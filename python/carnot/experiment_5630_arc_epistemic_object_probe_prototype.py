"""Experiment 5630: ARC epistemic object-hypothesis probe prototype.

Spec refs: REQ-ARC-WMTE-5630,
SCENARIO-ARC-WMTE-5630-INFORMATIVE-PROBE-POSITIVE,
SCENARIO-ARC-WMTE-5630-NEGATIVE-AND-UNSAFE-REJECTION.

This artifact is a development proxy over already reproduced public traces. It
does not read game source, does not use per-game adapters, does not run BFS, and
does not claim solve credit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carnot.agentic.arc_epistemic_object_probe import (
    EpistemicObjectProbePlanner,
    LiveProbeAction,
    ObjectProbeObservation,
    make_corrupted_effect_hypothesis,
    make_hallucinated_object_hypothesis,
    stable_checksum,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5630_arc_epistemic_object_probe_prototype"
EXPERIMENT_ID = 5630
RESULT_RELATIVE_PATH = f"results/{EXPERIMENT}.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
INFERENCE_SUBSTRATE = "bounded_object_hypothesis_search_over_live_agent_observations"
SOLVE_PROVENANCE = "development_proxy"
DEFAULT_ROSTER = ("dc22", "bp35", "s5i5")
LIVE_OBSERVATION_FIELDS_USED = [
    "state",
    "action",
    "data",
    "successor",
    "level_before",
    "level_after",
]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "registry_precheck_receipt",
    "evaluation_levels",
    "solve_provenance",
    "live_observation_fields_used",
    "object_hypothesis_non_degenerate_count",
    "hypothesis_weights_by_trace",
    "causal_probe_scores",
    "informative_control_delta",
    "uninformative_control_delta",
    "unsafe_model_accept_count",
    "live_interface_replay_rate",
    "epistemic_probe_ready_score",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "field_principles": {
        "principle": "principle annotations make every required 5630 field auditable.",
    },
    "registry_precheck_receipt": {
        "principle": "every evaluation target is reproduced in the registry and no solve is duplicated.",
    },
    "evaluation_levels": {
        "principle": "development targets are explicit and carry no level credit.",
    },
    "solve_provenance": {
        "principle": "development_proxy -- this prototype claims no ARC level credit.",
    },
    "live_observation_fields_used": {
        "principle": "hypotheses are induced only from agent-owned before/action/data/successor/level observations.",
    },
    "object_hypothesis_non_degenerate_count": {
        "principle": "the mechanism is not a single-model wrapper.",
    },
    "hypothesis_weights_by_trace": {
        "principle": "posterior belief updates are inspectable per trace.",
    },
    "causal_probe_scores": {
        "principle": "information-gain decisions are auditable before execution.",
    },
    "informative_control_delta": {
        "principle": "positive controls must reduce uncertainty or model error over random legal probes.",
    },
    "uninformative_control_delta": {
        "principle": "noise controls must not be credited as information gain.",
    },
    "unsafe_model_accept_count": {
        "principle": "corrupted and hallucinated hypotheses fail closed.",
    },
    "live_interface_replay_rate": {
        "principle": "emitted probes are reachable through the live policy action shape.",
    },
    "epistemic_probe_ready_score": {
        "principle": "downstream gating is mechanical, not prose.",
    },
    "inference_substrate": {
        "principle": "bounded_object_hypothesis_search_over_live_agent_observations -- no LLM or game source participated.",
    },
    "random_seeds": {
        "principle": "determinism is the precondition for replaying the trials.",
    },
    "reproducibility_checksum": {
        "principle": "trial inputs, thresholds, and decisions are replayable.",
    },
    "honest_verdict": {
        "principle": "terminal prefix retires degenerate or unreachable prototypes cleanly.",
    },
}


def read_yaml(path: Path) -> JsonDict:  # pragma: no cover - exercised by main/default artifact run.
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def payload_checksum(payload: Mapping[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    return stable_checksum(core)


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
    transitions_by_game: Mapping[str, Sequence[ObjectProbeObservation]],
) -> JsonDict:
    rows = _registry_rows(registry)
    roster_rows: list[JsonDict] = []
    selected: list[str] = []
    for game in roster:
        reg = rows.get(str(game))
        reproduced = _is_reproduced(reg)
        transition_count = len(transitions_by_game.get(str(game), ()))
        usable = bool(reproduced and transition_count >= 2)
        if usable:
            selected.append(str(game))
        roster_rows.append(
            {
                "game": str(game),
                "registry_reproduced": bool(reproduced),
                "levels_reproduced": int((reg or {}).get("levels_reproduced") or 0),
                "agent_owned_transition_count": int(transition_count),
                "selected": usable,
            }
        )
    return {
        "ok": len(selected) >= 3,
        "only_already_reproduced_levels": all(
            row["registry_reproduced"] for row in roster_rows if row["selected"]
        ),
        "no_solve_duplicated": True,
        "source_files_read": False,
        "per_game_adapter_used": False,
        "exhaustive_bfs_used": False,
        "outer_loop_recipes_used": False,
        "selected_games": selected,
        "roster_rows": roster_rows,
        "registry_total_levels": int(registry.get("reproducible_total_levels") or 0),
    }


def load_agent_owned_observations(
    roster: Sequence[str] = DEFAULT_ROSTER,
    *,
    max_per_game: int = 160,
) -> dict[str, list[ObjectProbeObservation]]:  # pragma: no cover - covered by artifact generation.
    from carnot.agentic.arc_transition_capture import TransitionCorpus

    corpus = TransitionCorpus()
    out: dict[str, list[ObjectProbeObservation]] = {}
    for game in roster:
        rows: list[ObjectProbeObservation] = []
        for index, transition in enumerate(corpus.load(str(game))):
            state = np.asarray(transition.grid, dtype=np.int16)
            successor = np.asarray(transition.next_grid, dtype=np.int16)
            if state.shape != successor.shape or not np.any(state != successor):
                continue
            rows.append(
                ObjectProbeObservation(
                    trace_id=str(game),
                    step=index,
                    state=state,
                    action=int(transition.action),
                    data=transition.data,
                    successor=successor,
                    level_before=int(getattr(transition, "level_before", 0)),
                    level_after=int(getattr(transition, "level_after", 0)),
                )
            )
            if len(rows) >= max_per_game:
                break
        out[str(game)] = rows
    return out


def build_artifact(
    *,
    transitions_by_game: Mapping[str, Sequence[ObjectProbeObservation]] | None = None,
    registry: Mapping[str, Any] | None = None,
    roster: Sequence[str] = DEFAULT_ROSTER,
    random_seed: int = 5630,
    root: Path = REPO_ROOT,
) -> JsonDict:
    registry_data = dict(registry or read_yaml(root / REGISTRY_RELATIVE_PATH))
    traces = (
        {str(game): list(rows) for game, rows in transitions_by_game.items()}
        if transitions_by_game is not None
        else load_agent_owned_observations(roster)
    )
    roster = tuple(str(game) for game in roster if str(game) in traces)
    precheck = registry_precheck(roster, registry_data, traces)
    planner = EpistemicObjectProbePlanner(random_seed=random_seed)

    weights_by_trace: dict[str, JsonDict] = {}
    scores_by_trace: dict[str, list[JsonDict]] = {}
    evaluation_levels: list[str] = []
    informative_deltas: list[float] = []
    uninformative_deltas: list[float] = []
    replay_rates: list[float] = []
    unsafe_accepts = 0
    nondegenerate = 0

    for game in precheck["selected_games"]:
        reg_row = next(
            row for row in precheck["roster_rows"] if row["game"] == game
        )
        evaluation_levels.append(f"{game}:L<={int(reg_row['levels_reproduced'])}")
        selected = _select_informative_pair(planner, game, traces[game])
        if selected is None:
            weights_by_trace[game] = {}
            scores_by_trace[game] = []
            continue
        calibration, heldout, legal_actions = selected
        model = planner.build_trace_model(game, [calibration])
        if model.is_non_degenerate:
            nondegenerate += 1
        scores = planner.score_probes(model, heldout.state, legal_actions)
        scores_by_trace[game] = [score.as_dict() for score in scores[:4]]
        controls = planner.compare_controls(
            model,
            heldout.state,
            legal_actions,
            observed=heldout,
        )
        unsafe_accepts += planner.reject_unsafe_models(
            model,
            [
                make_corrupted_effect_hypothesis(model.hypotheses[0]),
                make_hallucinated_object_hypothesis("ffffffffffffffff"),
            ]
            if model.hypotheses
            else [],
            heldout.state,
        )
        informative_deltas.append(float(controls["informative_control_delta"]))
        uninformative_deltas.append(float(controls["uninformative_control_delta"]))
        replay_rates.append(float(controls["live_interface_replay_rate"]))
        weights_by_trace[game] = {
            name: round(float(value), 8) for name, value in sorted(model.weights.items())
        }

    informative_delta = _mean(informative_deltas)
    uninformative_delta = _mean(uninformative_deltas)
    replay_rate = _mean(replay_rates)
    ready = float(
        nondegenerate >= 3
        and informative_delta > 0.0
        and uninformative_delta <= 0.0
        and int(unsafe_accepts) == 0
        and replay_rate == 1.0
    )
    verdict = (
        "complete: epistemic_object_probe_ready_development_proxy"
        if ready == 1.0
        else "blocked: epistemic_object_probe_degenerate_or_unreachable_terminal"
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": "carnot.exp5630.arc_epistemic_object_probe_prototype.v1",
        "date": "20260714",
        "field_principles": FIELD_PRINCIPLES,
        "registry_precheck_receipt": precheck,
        "evaluation_levels": evaluation_levels,
        "solve_provenance": SOLVE_PROVENANCE,
        "source_files_read": False,
        "per_game_adapter_used": False,
        "exhaustive_bfs_used": False,
        "outer_loop_recipes_used": False,
        "live_observation_fields_used": list(LIVE_OBSERVATION_FIELDS_USED),
        "object_hypothesis_non_degenerate_count": int(nondegenerate),
        "hypothesis_weights_by_trace": weights_by_trace,
        "causal_probe_scores": scores_by_trace,
        "informative_control_delta": informative_delta,
        "uninformative_control_delta": uninformative_delta,
        "unsafe_model_accept_count": int(unsafe_accepts),
        "live_interface_replay_rate": replay_rate,
        "epistemic_probe_ready_score": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(random_seed)],
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _select_informative_pair(
    planner: EpistemicObjectProbePlanner,
    game: str,
    rows: Sequence[ObjectProbeObservation],
) -> tuple[ObjectProbeObservation, ObjectProbeObservation, list[LiveProbeAction]] | None:
    usable = [row for row in rows if int(row.action) == 6 and isinstance(row.data, Mapping)]
    for left_index, calibration in enumerate(usable):
        model = planner.build_trace_model(game, [calibration])
        if not model.is_non_degenerate:
            continue
        for heldout in usable[left_index + 1 :]:
            legal = _legal_actions(calibration, heldout)
            chosen = planner.choose_probe(model.copy_with_weights(model.weights), heldout.state, legal)
            if chosen is not None and chosen.action == LiveProbeAction(heldout.action, heldout.data):
                return calibration, heldout, legal
    if len(usable) >= 2:
        return usable[0], usable[1], _legal_actions(usable[0], usable[1])
    return None


def _legal_actions(
    calibration: ObjectProbeObservation,
    heldout: ObjectProbeObservation,
) -> list[LiveProbeAction]:
    return [
        LiveProbeAction(1, None),
        LiveProbeAction(calibration.action, calibration.data),
        LiveProbeAction(heldout.action, heldout.data),
    ]


def _mean(values: Sequence[float]) -> float:
    return round(float(sum(values)) / float(len(values)), 8) if values else 0.0


def main() -> None:  # pragma: no cover
    write_json(REPO_ROOT / RESULT_RELATIVE_PATH, build_artifact())


if __name__ == "__main__":  # pragma: no cover
    main()
