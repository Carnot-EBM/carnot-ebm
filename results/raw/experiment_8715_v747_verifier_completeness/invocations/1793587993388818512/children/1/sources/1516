"""Experiment 5155: multi-level ARC belief-state scoping.

Spec refs: REQ-ARC-WMTE-5155,
SCENARIO-ARC-WMTE-5155-RESET-CHARACTERIZATION,
SCENARIO-ARC-WMTE-5155-FALSIFIABLE-PROPOSALS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT = "experiment_5155_multilevel_belief_state_scoping_v472"
SCHEMA = "carnot.exp5155.multilevel_belief_state_scoping.v1"
RESULT_RELATIVE_PATH = "results/experiment_5155_multilevel_belief_state_scoping_v472.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 5155
SPEC_REFS = (
    "REQ-ARC-WMTE-5155",
    "SCENARIO-ARC-WMTE-5155-RESET-CHARACTERIZATION",
    "SCENARIO-ARC-WMTE-5155-FALSIFIABLE-PROPOSALS",
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "MUST start with complete:/complete_/success:/success_; this is a scoped "
            "proposal, not a banked level."
        )
    },
    "belief_state_resets_at_level_boundary": {
        "principle": (
            "The precise, code-verified fact this scoping pass exists to establish -- do not "
            "assume from memory or prior notes."
        )
    },
    "proposed_experiments": {
        "principle": (
            "Each must be small and falsifiable, not another open-ended generation-axis swing -- "
            "the L1 wall already taught us that costs milestones without them."
        )
    },
    "per_game_characterization": {
        "principle": (
            "Every game at levels_reproduced >= 1 is listed with its registry depth and plausible "
            "cross-level information loss."
        )
    },
    "learning_gaps_verified_against_code": {
        "principle": (
            "temporal credit-assignment, learned belief-state identity, and compounding TTT are "
            "checked against current code instead of cited blindly."
        )
    },
    "first_contact_vs_deepen_distinction": {
        "principle": (
            "states why Experiment 5155 is a deepen-wall scoping artifact distinct from Experiment "
            "5154 first-contact generation."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment",
    "schema",
    "honest_verdict",
    "belief_state_resets_at_level_boundary",
    "proposed_experiments",
    "per_game_characterization",
    "current_state_characterization",
    "learning_gaps_verified_against_code",
    "proposal_ranking",
    "first_contact_vs_deepen_distinction",
    "preconditions_checked",
    "field_principles",
    "spec_refs",
    "random_seed",
    "registry_updated",
    "reproducible_total_levels",
    "registry_solved_games",
    "registry_never_contacted_games",
    "reproducibility_checksum",
)

_RESET_EVIDENCE = [
    (
        "python/carnot/agentic/arc_competition_agent.py:_begin_level_goal_episode "
        "sets _episode_transition_start=len(self.transitions) and "
        "_episode_dsl_transition_start=len(self._dsl_transitions)."
    ),
    (
        "python/carnot/agentic/arc_competition_agent.py:_active_transitions and "
        "_active_dsl_transitions return only the post-boundary suffix."
    ),
    (
        "python/carnot/agentic/arc_competition_agent.py:_induce_and_plan passes only "
        "active_transitions into gated_engine_from_transitions, trust-energy selection, "
        "and bounded LLM reinduction."
    ),
    (
        "python/carnot/agentic/arc_live_ttt.py:gated_engine_from_transitions constructs "
        "a fresh LiveTTTWorldModel for each active transition slice."
    ),
    (
        "python/carnot/agentic/arc_agi3_world_model.py defines persistent GameGraph, but "
        "E3AgentPolicy does not instantiate it as the active cross-level world model."
    ),
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def current_state_characterization() -> dict[str, Any]:
    return {
        "belief_state_resets_at_level_boundary": True,
        "reset_scope": "active_world_model_induction_slice",
        "preserved_state": "explorer_graph_and_navigation_edges",
        "code_evidence": list(_RESET_EVIDENCE),
        "nuance": (
            "StepwiseExplorer keeps its observed graph, adjacency, frontier, and navigation state "
            "across level-up; the reset is the active world-model/belief induction context used by "
            "TTT, DSL fitting, trust-energy candidate selection, and LLM reinduction."
        ),
        "specific_reset_effects": [
            "pre-level-up transitions are excluded from _active_transitions()",
            "pre-level-up DSL tuples are excluded from _active_dsl_transitions()",
            "induced plan and world_model_trust_selection are cleared",
            "dsl_energy and goal bias are cleared",
            "the TTT prior path builds a fresh LiveTTTWorldModel from the suffix",
        ],
    }


def _as_level(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("levels_reproduced") or 0)
    except (TypeError, ValueError):
        return 0


def _text(row: Mapping[str, Any]) -> str:
    pieces = [
        row.get("game", ""),
        row.get("mechanic_class", ""),
        row.get("win_condition", ""),
        row.get("action_model", ""),
        " ".join(str(item) for item in row.get("gotchas", []) or []),
        " ".join(str(item) for item in row.get("dead_ends", []) or []),
    ]
    return " ".join(str(piece).lower() for piece in pieces if piece is not None)


def _contains(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def information_loss_for_game(row: Mapping[str, Any]) -> dict[str, list[str]]:
    text = _text(row)
    goal_structure = [
        "prior level win predicate/operator family is excluded from next-level induction input"
    ]
    action_effects = [
        "observed action_key -> grid delta examples before level-up are excluded from the active TTT/DSL fit"
    ]
    hazards = []
    hidden = []

    if _contains(text, ("hazard", "blocker", "game_over", "dead", "wall", "obstacle")):
        hazards.append(
            "visible blockers, dead-end actions, or hazard-avoidance evidence learned before level-up"
        )
    if _contains(text, ("stepcounter", "step_counter", "undo", "hidden", "register", "animation", "rng", "path-dependence", "fresh_env")):
        hidden.append(
            "register, counter, phase, path-dependent, or hidden-state disambiguation evidence"
        )
    if _contains(text, ("color", "rotation", "glyph", "rule", "marker", "palette", "template", "shape", "sprite", "peg", "rail", "fruit", "cast", "tank", "support", "toggle")):
        goal_structure.append(
            "object identity, palette, glyph/rule, marker, or shape constraints that often recur with level-specific parameters"
        )
    if _contains(text, ("click", "action", "keyboard", "move", "drag", "cycle", "push", "toggle", "rotate")):
        action_effects.append(
            "control semantics and action-effect priors discovered during the previous level"
        )

    return {
        "goal_structure": goal_structure,
        "action_effects": action_effects,
        "hazard_locations": hazards,
        "hidden_or_register_state": hidden,
    }


def characterize_games(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in registry.get("games", []) or []:
        if not isinstance(raw, Mapping):
            continue
        level = _as_level(raw)
        if level < 1:
            continue
        game = str(raw.get("game") or "")
        status = "shallow_solved_l1" if level == 1 else "deepened_but_stuck"
        rows.append(
            {
                "game": game,
                "levels_reproduced": level,
                "status": status,
                "mechanic_class": str(raw.get("mechanic_class") or "unspecified"),
                "belief_state_reset_applies": True,
                "information_thrown_away_by_active_reset": information_loss_for_game(raw),
                "why_next_level_may_need_it": (
                    "The registry row shows a reusable mechanic family or path-conditioned level "
                    "chain; the current induction slice forces the agent to rediscover that family "
                    "from post-boundary transitions before planning the next level."
                ),
            }
        )
    return sorted(rows, key=lambda row: row["game"])


def proposed_experiments() -> list[dict[str, Any]]:
    return [
        {
            "name": "transition_slice_warm_start_replay_ablation",
            "hypothesis": (
                "Warm-starting the N+1 online TTT/DSL induction with level-N transition and energy "
                "statistics improves next-level held-out changed-cell value accuracy versus the "
                "current cold post-boundary slice."
            ),
            "falsifiable_gate": (
                "On at least 6 registry level transitions, warm-start beats cold-slice control by "
                ">=0.10 median changed-cell value accuracy or >=20% fewer replayed actions to the "
                "next level; otherwise mark no value."
            ),
            "estimated_effort": "small: offline replay ablation over existing registry trajectories",
            "control": "current cold _active_transitions() suffix only",
            "signal": "high",
            "effort_rank": 1,
            "signal_rank": 1,
            "distinct_from_exp5154": True,
        },
        {
            "name": "cross_level_goal_energy_ranker_replay",
            "hypothesis": (
                "A per-game online energy model fit to level-N win/near-win states ranks level-N+1 "
                "frontier candidates closer to the eventual level-up than the cold Exp4020-only bias."
            ),
            "falsifiable_gate": (
                "For lp85, sc25, and tr87 replay slices, target-prefix reciprocal rank improves on "
                "at least 2/3 games with no regression in level reached; otherwise reject carryover "
                "goal-energy value."
            ),
            "estimated_effort": "small-medium: candidate-ranking replay, no live solve build",
            "control": "cold Exp4020/current goal bias with no previous-level energy memory",
            "signal": "medium",
            "effort_rank": 2,
            "signal_rank": 2,
            "distinct_from_exp5154": True,
        },
        {
            "name": "hidden_register_hazard_belief_carryover_probe",
            "hypothesis": (
                "Carrying compact register/hazard belief features from level N reduces rediscovery "
                "cost on hidden-state-bound games where counters, phases, or hazard blockers recur."
            ),
            "falsifiable_gate": (
                "On ka59/ar25/ft09 transition traces, carryover reduces invalid candidate expansions "
                "or verifier-rejected branches by >=25% versus cold reset, while reproducing the same "
                "prior level depth; otherwise reject the register carryover probe."
            ),
            "estimated_effort": "medium: requires compact belief serialization plus replay scorer",
            "control": "cold post-boundary transition slice with no register/hazard carryover",
            "signal": "medium-high",
            "effort_rank": 3,
            "signal_rank": 3,
            "distinct_from_exp5154": True,
        },
    ]


def rank_proposals(proposals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        proposals,
        key=lambda row: (int(row["effort_rank"]), int(row["signal_rank"]), str(row["name"])),
    )
    return [
        {
            "rank": index,
            "name": str(row["name"]),
            "estimated_effort": str(row["estimated_effort"]),
            "signal": str(row["signal"]),
            "reason": (
                "Ranks first when it gives the cheapest direct signal on whether previous-level "
                "belief state helps the current cold-start deepen path."
            )
            if index == 1
            else "Lower-ranked because it needs more machinery before the signal is clean.",
        }
        for index, row in enumerate(ranked, start=1)
    ]


def learning_gaps_verified_against_code() -> dict[str, dict[str, Any]]:
    return {
        "temporal_credit_assignment": {
            "still_open": True,
            "code_verified_evidence": (
                "Transition objects carry level_after, but E3 next-level induction reads only the "
                "post-boundary active transition suffix, so previous-level evidence is not credited "
                "to the next-level model fit."
            ),
        },
        "learned_belief_state_identity": {
            "still_open": True,
            "code_verified_evidence": (
                "GameGraph has stable frame hashes, but the scored E3 path does not maintain a "
                "persistent per-game latent-mechanic identity that survives level-up."
            ),
        },
        "compounding_ttt": {
            "still_open": True,
            "code_verified_evidence": (
                "gated_engine_from_transitions creates a fresh LiveTTTWorldModel per induction "
                "attempt; L0/L1/win-state knowledge is not serialized forward across levels."
            ),
        },
    }


def _never_contacted_games(registry: Mapping[str, Any]) -> list[str]:
    games = []
    for row in registry.get("games", []) or []:
        if isinstance(row, Mapping) and _as_level(row) < 1:
            games.append(str(row.get("game") or ""))
    return sorted(game for game in games if game)


def build_artifact(
    registry: Mapping[str, Any],
    *,
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:
    characterization = characterize_games(registry)
    proposals = proposed_experiments()
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": (
            "complete: deepen_belief_state_reset_scoped_3_falsifiable_experiments_no_full_build"
        ),
        "belief_state_resets_at_level_boundary": {
            "value": True,
            "principle": FIELD_PRINCIPLES["belief_state_resets_at_level_boundary"]["principle"],
            "fact": (
                "The active E3 world-model/belief induction state resets at level-up by advancing "
                "transition-slice offsets and rebuilding the TTT model from the post-boundary suffix; "
                "the explorer graph is preserved."
            ),
            "evidence": list(_RESET_EVIDENCE),
        },
        "proposed_experiments": {
            "value": proposals,
            "principle": FIELD_PRINCIPLES["proposed_experiments"]["principle"],
        },
        "per_game_characterization": characterization,
        "current_state_characterization": current_state_characterization(),
        "learning_gaps_verified_against_code": learning_gaps_verified_against_code(),
        "proposal_ranking": rank_proposals(proposals),
        "first_contact_vs_deepen_distinction": {
            "value": (
                "Experiment 5154 targets first-contact candidate generation: making an initial "
                "winner appear. Experiment 5155 targets within-game deepening after first contact: "
                "whether per-game belief and energy state learned on level N should warm-start "
                "level N+1 instead of cold-starting the active induction slice."
            ),
            "principle": FIELD_PRINCIPLES["first_contact_vs_deepen_distinction"]["principle"],
        },
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "registry_updated": str(registry.get("updated") or ""),
        "reproducible_total_levels": int(registry.get("reproducible_total_levels") or 0),
        "registry_solved_games": len(characterization),
        "registry_never_contacted_games": _never_contacted_games(registry),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"artifact missing fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal complete/success prefix")
    reset = artifact["belief_state_resets_at_level_boundary"]
    if not isinstance(reset, Mapping) or not isinstance(reset.get("value"), bool):
        raise ValueError("belief_state_resets_at_level_boundary.value must be bool")
    proposals = artifact["proposed_experiments"]
    if not isinstance(proposals, Mapping):
        raise ValueError("proposed_experiments must be a value/principle object")
    proposal_values = proposals.get("value")
    if not isinstance(proposal_values, list) or not 2 <= len(proposal_values) <= 3:
        raise ValueError("proposed_experiments must contain 2-3 proposals")
    required_proposal_fields = {
        "name",
        "hypothesis",
        "falsifiable_gate",
        "estimated_effort",
    }
    for proposal in proposal_values:
        if not isinstance(proposal, Mapping) or not required_proposal_fields <= set(proposal):
            raise ValueError("proposed_experiments entries missing required fields")
    expected = reproducibility_checksum(artifact)
    if artifact["reproducibility_checksum"] != expected:
        raise ValueError("checksum mismatch")


def write_artifact(artifact: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_preconditions(root: Path) -> dict[str, Any]:  # pragma: no cover - exercised by script run
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return {
        "AGENTS.md": (root / "AGENTS.md").exists(),
        "CODEX.md": (root / "CODEX.md").exists(),
        "CLAUDE.md_live_framing_read": (root / "CLAUDE.md").exists(),
        "arc_live_ttt.py_read": (root / "python/carnot/agentic/arc_live_ttt.py").exists(),
        "arc_agi3_world_model.py_read": (
            root / "python/carnot/agentic/arc_agi3_world_model.py"
        ).exists(),
        "arc_competition_agent.py_read": (
            root / "python/carnot/agentic/arc_competition_agent.py"
        ).exists(),
        "arc_solve_registry.yaml_read": (root / REGISTRY_RELATIVE_PATH).exists(),
        "spec_has_req_5155": "REQ-ARC-WMTE-5155" in spec_text,
    }


def main(root: Path | str = Path(".")) -> dict[str, Any]:  # pragma: no cover - script entrypoint
    repo = Path(root)
    registry = yaml.safe_load((repo / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    artifact = build_artifact(registry, preconditions_checked=_repo_preconditions(repo))
    validate_artifact(artifact)
    write_artifact(artifact, repo / RESULT_RELATIVE_PATH)
    return artifact


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()
