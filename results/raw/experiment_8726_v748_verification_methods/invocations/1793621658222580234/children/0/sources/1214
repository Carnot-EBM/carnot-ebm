"""Experiment 4821: S3 structural-energy generation lift.

Spec refs: REQ-ARC-WMTE-4821,
SCENARIO-ARC-WMTE-4821-GENERATION-LIFT,
SCENARIO-ARC-WMTE-4821-LIVE-PLAN-WIRING.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4821_structural_energy_s3_generation_lift"
EXPERIMENT_ID = 4821
SCHEMA = "carnot.arc_structural_energy_s3_generation_lift_4821.v1"
RESULT_RELATIVE_PATH = "results/experiment_4821_structural_energy_s3_generation_lift.json"
S2V3_RELATIVE_PATH = "results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json"
GENERATION_MEASUREMENT_RELATIVE_PATH = (
    "results/experiment_4737_goal_energy_candidate_generation_valid_test.json"
)
POSITIVE_CONTROL_RELATIVE_PATH = "results/experiment_4640_goal_energy_generation_live.json"
R11L_RANK_RELATIVE_PATH = "results/experiment_4700_object_centric_perception_proposal_live.json"
RANDOM_SEED = 4821
BOOTSTRAP_RESAMPLES = 1000
MIN_HEADROOM_GAMES = 5
GUIDANCE_LAMBDA = 1.0
CONTROL_LAMBDA = 0.0
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "live_agent_self_discovery"

SUCCESS_VERDICT = "success_structural_energy_s3_generation_authorizes_s4"
BOUNDED_VERDICT = "complete_structural_energy_s3_bounded_no_generation_lift"
INCONCLUSIVE_VERDICT = "complete_structural_energy_s3_inconclusive_no_generation_headroom"
TERMINAL_PREFIXES = ("success_", "success:", "complete_", "complete:", "blocked_")

SPEC_REFS = [
    "REQ-ARC-WMTE-4821",
    "SCENARIO-ARC-WMTE-4821-GENERATION-LIFT",
    "SCENARIO-ARC-WMTE-4821-LIVE-PLAN-WIRING",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; a generation win is success_structural_energy_s3_generation_authorizes_s4, a null is complete_structural_energy_s3_bounded_no_generation_lift, insufficient headroom is complete_structural_energy_s3_inconclusive_no_generation_headroom."
    },
    "verifier_is_oracle": {
        "principle": "MUST be false -- the energy guides generation, oracle-distinct; required for check_circular_moat_overclaim."
    },
    "live_path_reachable": {
        "principle": "the goal_energy guidance must be in the E3AgentPolicy plan_in_model import closure (arc_orphan_solver_lint) -- a guidance the live agent cannot reach adds no live value."
    },
    "inference_substrate": {
        "principle": "verifier_ensemble_against_cached_candidates (the energy guides over cached candidates; if the live LLM induces fresh, declare live_llm_inference)."
    },
    "n_headroom_games": {
        "principle": ">=5 held-out games where the bare explorer banks 0 AND the winner is reachable -- the ONLY games that test generation (positive control: generation COULD help)."
    },
    "winners_newly_entering_pool_delta_ci95": {
        "principle": "the decisive metric -- E-guided minus lambda=0 fraction of winners NEWLY entering the pool; must EXCLUDE 0 for a PASS."
    },
    "new_levels_not_in_bare_pool": {
        "principle": "any banked level must NOT have been already in the bare explorer's pool -- else it is re-ranking, not generation (the second killer)."
    },
    "solve_provenance": {
        "principle": "live_agent_self_discovery -- a generation lift is the agent solving via its OWN E-guided attempts; offline_reproduced gated."
    },
    "game_results": {
        "principle": "per-game: winner-rank, banked-by-E, banked-by-bare, was-already-in-bare-pool -- the per-game generation log."
    },
    "positive_control_passed": {
        "principle": "the headroom games' winners ARE reachable -- so a null means 'energy could have generated the winner but didn't', not 'no winner existed'."
    },
    "random_seed": {"principle": "determinism for the guided/unguided runs + bootstrap."},
    "reproducibility_checksum": {
        "principle": "content hash of (energy, games, lambda, seeds) so a replication catches drift."
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "verifier_is_oracle",
    "live_path_reachable",
    "inference_substrate",
    "preconditions_checked",
    "n_headroom_games",
    "min_headroom_games",
    "winners_newly_entering_pool_delta",
    "winners_newly_entering_pool_delta_ci95",
    "new_levels_not_in_bare_pool",
    "solve_provenance",
    "game_results",
    "positive_control_passed",
    "retire_if_same_verdict",
    "lambda_guidance",
    "lambda0_control",
    "energy_config",
    "source_artifacts",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _attempted_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [dict(row) for row in rows if row.get("attempted", True) is True]


def _rows_by_game(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = {}
    for row in _attempted_rows(rows):
        game = str(row.get("game") or "")
        if not game:
            continue
        grouped.setdefault(game, []).append(dict(row))
    return grouped


def _offline_reproduced(row: Mapping[str, Any]) -> bool:
    gate = row.get("reproduction_gate")
    return bool(
        row.get("offline_reproduced") is True
        or (isinstance(gate, Mapping) and gate.get("reproduced") is True)
    )


def _reached_level(row: Mapping[str, Any]) -> int:
    gate = row.get("reproduction_gate")
    candidates = [row.get("reached_level"), row.get("reproduced_levels")]
    if isinstance(gate, Mapping):
        candidates.extend([gate.get("reached_level"), gate.get("claimed_level")])
    out = 0
    for value in candidates:
        try:
            out = max(out, int(value or 0))
        except (TypeError, ValueError):
            continue
    return out


def _banked(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(_offline_reproduced(row) and _reached_level(row) > 0 for row in rows)


def _max_reached(rows: Sequence[Mapping[str, Any]]) -> int:
    return max((_reached_level(row) for row in rows), default=0)


def _labels(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        for label in row.get("solution_labels") or []:
            out.add(_stable_json(label))
    return out


def _positive_reachable(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        row.get("reachable_headroom") is True
        or row.get("positive_control_reachable") is True
        or row.get("bare_control_passed") is True
        for row in rows
    )


def _winner_rank_for(game: str, winner_rank_by_game: Mapping[str, Any] | None) -> JsonDict:
    if winner_rank_by_game and game in winner_rank_by_game:
        value = winner_rank_by_game[game]
        return dict(value) if isinstance(value, Mapping) else {"value": value}
    return {"rank": None, "candidate_count": None, "source": "not_measured_for_cached_game"}


def generation_lift_rows(
    *,
    bare_attempts: Sequence[Mapping[str, Any]],
    guided_attempts: Sequence[Mapping[str, Any]],
    positive_control_attempts: Sequence[Mapping[str, Any]] | None = None,
    winner_rank_by_game: Mapping[str, Any] | None = None,
    selected_games: Sequence[str] | None = None,
) -> list[JsonDict]:
    """REQ-ARC-WMTE-4821: build per-game matched generation rows."""

    bare_by_game = _rows_by_game(bare_attempts)
    guided_by_game = _rows_by_game(guided_attempts)
    positive_by_game = _rows_by_game(positive_control_attempts or [])
    games = sorted(set(bare_by_game) & set(guided_by_game))
    if selected_games is not None:
        allowed = {str(game) for game in selected_games}
        games = [game for game in games if game in allowed]

    out: list[JsonDict] = []
    for game in games:
        bare_rows = bare_by_game.get(game, [])
        guided_rows = guided_by_game.get(game, [])
        positive_rows = positive_by_game.get(game, [])
        bare_banked = _banked(bare_rows)
        guided_banked = _banked(guided_rows)
        bare_labels = _labels(bare_rows)
        guided_labels = _labels(guided_rows)
        already_in_bare = bool(guided_banked and bare_banked and (not guided_labels or bare_labels & guided_labels))
        newly_entered = bool(guided_banked and not bare_banked and not already_in_bare)
        winner_rank = _winner_rank_for(game, winner_rank_by_game)
        row = {
            "game": game,
            "winner_rank": winner_rank,
            "winner-rank": winner_rank,
            "banked_by_E": bool(guided_banked),
            "banked-by-E": bool(guided_banked),
            "banked_by_bare": bool(bare_banked),
            "banked-by-bare": bool(bare_banked),
            "was_already_in_bare_pool": bool(already_in_bare),
            "was-already-in-bare-pool": bool(already_in_bare),
            "winner_newly_entered_pool": bool(newly_entered),
            "positive_control_reachable": _positive_reachable(positive_rows),
            "bare_reached_level": _max_reached(bare_rows),
            "e_guided_reached_level": _max_reached(guided_rows),
            "bare_offline_reproduced": bool(bare_banked),
            "e_guided_offline_reproduced": bool(guided_banked),
            "lambda0_attempts": len(bare_rows),
            "e_guided_attempts": len(guided_rows),
        }
        out.append(row)
    return out


def _headroom_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        dict(row)
        for row in rows
        if row.get("positive_control_reachable") is True and row.get("banked_by_bare") is False
    ]


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    if not values:
        return None
    vals = [float(value) for value in values]
    if len(set(vals)) == 1:
        value = round(vals[0], 6)
        return [value, value]
    rng = random.Random(int(seed))
    means: list[float] = []
    for _ in range(max(1, int(resamples))):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / float(len(sample)))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return [round(float(lo), 6), round(float(hi), 6)]


def _ci_excludes_zero(ci95: Sequence[float] | None) -> bool:
    return bool(ci95 is not None and len(ci95) == 2 and float(ci95[0]) > 0.0)


def _new_levels_not_in_bare_pool(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "game": str(row["game"]),
            "e_guided_reached_level": int(row.get("e_guided_reached_level", 0) or 0),
            "bare_reached_level": int(row.get("bare_reached_level", 0) or 0),
            "offline_reproduced": bool(row.get("e_guided_offline_reproduced")),
        }
        for row in rows
        if row.get("winner_newly_entered_pool") is True
    ]


def _verdict(
    *,
    n_headroom: int,
    positive_control_passed: bool,
    live_path_reachable: bool,
    ci95: Sequence[float] | None,
    new_levels: Sequence[Mapping[str, Any]],
) -> str:
    if n_headroom < MIN_HEADROOM_GAMES:
        return INCONCLUSIVE_VERDICT
    if (
        positive_control_passed
        and live_path_reachable
        and new_levels
        and _ci_excludes_zero(ci95)
    ):
        return SUCCESS_VERDICT
    return BOUNDED_VERDICT if positive_control_passed else INCONCLUSIVE_VERDICT


def build_artifact(
    game_results: Sequence[Mapping[str, Any]],
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    lambda_guidance: float = GUIDANCE_LAMBDA,
    duration_s: float = 1.0,
    energy_config: Mapping[str, Any] | None = None,
    source_artifacts: Mapping[str, Any] | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4821-GENERATION-LIFT: assemble the S3 verdict."""

    rows = [dict(row) for row in game_results]
    headroom = _headroom_rows(rows)
    sufficient = len(headroom) >= MIN_HEADROOM_GAMES
    values = [1.0 if row.get("winner_newly_entered_pool") is True else 0.0 for row in headroom]
    delta = round(sum(values) / float(len(values)), 6) if sufficient and values else None
    ci95 = (
        _bootstrap_mean_ci(values, seed=int(random_seed), resamples=bootstrap_resamples)
        if sufficient
        else None
    )
    positive_control = bool(sufficient and all(row.get("positive_control_reachable") is True for row in headroom))
    new_levels = _new_levels_not_in_bare_pool(headroom)
    verdict = _verdict(
        n_headroom=len(headroom),
        positive_control_passed=positive_control,
        live_path_reachable=bool(live_path_reachable),
        ci95=ci95,
        new_levels=new_levels,
    )
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "n_headroom_games": len(headroom),
        "min_headroom_games": MIN_HEADROOM_GAMES,
        "winners_newly_entering_pool_delta": delta,
        "winners_newly_entering_pool_delta_ci95": ci95,
        "new_levels_not_in_bare_pool": new_levels,
        "solve_provenance": SOLVE_PROVENANCE,
        "game_results": rows,
        "positive_control_passed": bool(positive_control),
        "retire_if_same_verdict": True,
        "lambda_guidance": float(lambda_guidance),
        "lambda0_control": {
            "lambda": CONTROL_LAMBDA,
            "matched_control": True,
            "description": "goal_guidance_lambda=0 disables plan_in_model goal_energy guidance",
        },
        "energy_config": dict(energy_config or {}),
        "source_artifacts": dict(source_artifacts or {}),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(float(duration_s), 3),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-ARC-WMTE-4821: validate the falsifiable S3 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require(artifact["verifier_is_oracle"] is False, "verifier_is_oracle must be false")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact["solve_provenance"] == SOLVE_PROVENANCE, "solve_provenance")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["reproducibility_checksum"] == payload_checksum(artifact), "checksum mismatch")
    rows = [dict(row) for row in artifact["game_results"]]
    for row in rows:
        _require(row.get("winner_rank") == row.get("winner-rank"), "winner_rank alias mismatch")
        _require(row.get("banked_by_E") == row.get("banked-by-E"), "banked_by_E alias mismatch")
        _require(
            row.get("banked_by_bare") == row.get("banked-by-bare"),
            "banked_by_bare alias mismatch",
        )
        _require(
            row.get("was_already_in_bare_pool") == row.get("was-already-in-bare-pool"),
            "bare-pool alias mismatch",
        )
    headroom = _headroom_rows(rows)
    _require(artifact["n_headroom_games"] == len(headroom), "n_headroom_games mismatch")
    sufficient = len(headroom) >= MIN_HEADROOM_GAMES
    if not sufficient:
        _require(artifact["honest_verdict"] == INCONCLUSIVE_VERDICT, "insufficient headroom verdict")
        _require(artifact["winners_newly_entering_pool_delta_ci95"] is None, "inconclusive ci")
        return
    _require(artifact["positive_control_passed"] is True, "positive control required")
    if artifact["honest_verdict"] == SUCCESS_VERDICT:
        _require(
            artifact["new_levels_not_in_bare_pool"]
            and _ci_excludes_zero(artifact["winners_newly_entering_pool_delta_ci95"])
            and artifact["live_path_reachable"] is True,
            "success requires E-only offline-reproduced levels and CI95 excluding zero",
        )
    elif artifact["honest_verdict"] == BOUNDED_VERDICT:
        _require(artifact["retire_if_same_verdict"] is True, "bounded retire flag")
    else:
        raise ValueError("unexpected verdict for sufficient headroom")


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _r11l_winner_rank(root: Path) -> dict[str, Any]:
    path = root / R11L_RANK_RELATIVE_PATH
    if not path.exists():
        return {}
    coverage = _read_json(path).get("proposal_coverage_by_representation", {})
    object_rows = coverage.get("object_centric", {}) if isinstance(coverage, Mapping) else {}
    hits = object_rows.get("step_hits", []) if isinstance(object_rows, Mapping) else []
    if not hits:
        return {}
    return {
        "r11l": {
            "ranks": [hit.get("rank") for hit in hits],
            "candidate_counts": [hit.get("candidate_count") for hit in hits],
            "source": R11L_RANK_RELATIVE_PATH,
        }
    }


def _default_energy_config(root: Path) -> JsonDict:
    path = root / S2V3_RELATIVE_PATH
    if not path.exists():
        return {}
    artifact = _read_json(path)
    config = dict(artifact.get("energy_config") or {})
    config["s2v3_honest_verdict"] = artifact.get("honest_verdict")
    config["s2v3_energy_minus_accuracy_delta_ci95"] = artifact.get(
        "energy_minus_accuracy_delta_ci95"
    )
    return config


def _selected_headroom_games(root: Path) -> list[str]:
    control = _read_json(root / POSITIVE_CONTROL_RELATIVE_PATH)
    rows = control.get("baseline_measurement", {}).get("variant_attempts", [])
    return sorted(
        str(row["game"])
        for row in rows
        if row.get("reachable_headroom") is True and not (row.get("first_win") or row.get("solved"))
    )


def build_from_existing_artifacts(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    duration_s: float,
) -> JsonDict:
    """REQ-ARC-WMTE-4821: reproduce the S3 aggregate from cached matched attempts."""

    base = Path(root)
    generation = _read_json(base / GENERATION_MEASUREMENT_RELATIVE_PATH)
    positive_control = _read_json(base / POSITIVE_CONTROL_RELATIVE_PATH)
    selected = _selected_headroom_games(base)
    rows = generation_lift_rows(
        bare_attempts=generation["baseline_measurement"]["variant_attempts"],
        guided_attempts=generation["goal_energy_measurement"]["variant_attempts"],
        positive_control_attempts=positive_control["baseline_measurement"]["variant_attempts"],
        winner_rank_by_game=_r11l_winner_rank(base),
        selected_games=selected,
    )
    return build_artifact(
        rows,
        preconditions_checked=preconditions_checked,
        live_path_reachable=live_path_reachable,
        duration_s=duration_s,
        energy_config=_default_energy_config(base),
        source_artifacts={
            "s2v3_energy": S2V3_RELATIVE_PATH,
            "matched_generation_measurement": GENERATION_MEASUREMENT_RELATIVE_PATH,
            "positive_control": POSITIVE_CONTROL_RELATIVE_PATH,
            "winner_rank_diagnostic": R11L_RANK_RELATIVE_PATH,
        },
    )


def _command_ok(command: Sequence[str], *, root: Path) -> tuple[bool, str]:  # pragma: no cover
    try:
        completed = subprocess.run(
            list(command),
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except Exception as exc:
        return False, repr(exc)
    output = (completed.stdout + completed.stderr).strip()
    return completed.returncode == 0, output[-800:]


def _preconditions(root: Path) -> tuple[JsonDict, bool]:  # pragma: no cover
    py = str(root / ".venv" / "bin" / "python")
    offline_ok, offline_output = _command_ok(
        [py, "-c", "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()"],
        root=root,
    )
    e3_ok, e3_output = _command_ok(
        [py, "-c", "from carnot.agentic.arc_competition_agent import E3AgentPolicy"],
        root=root,
    )
    lint_ok, lint_output = _command_ok([py, "scripts/arc_orphan_solver_lint.py"], root=root)
    artifacts = {
        S2V3_RELATIVE_PATH: (root / S2V3_RELATIVE_PATH).exists(),
        GENERATION_MEASUREMENT_RELATIVE_PATH: (root / GENERATION_MEASUREMENT_RELATIVE_PATH).exists(),
        POSITIVE_CONTROL_RELATIVE_PATH: (root / POSITIVE_CONTROL_RELATIVE_PATH).exists(),
    }
    preconditions = {
        "offline_arcade": {"passed": offline_ok, "output_tail": offline_output},
        "e3_agent_policy_import": {"passed": e3_ok, "output_tail": e3_output},
        "arc_orphan_solver_lint": {"passed": lint_ok, "output_tail": lint_output},
        "source_artifacts_present": artifacts,
    }
    return preconditions, bool(offline_ok and e3_ok and lint_ok and all(artifacts.values()))


def run(*, root: Path | str = REPO_ROOT, write: bool = True) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    base = Path(root)
    preconditions, live_path_reachable = _preconditions(base)
    artifact = build_from_existing_artifacts(
        root=base,
        preconditions_checked=preconditions,
        live_path_reachable=live_path_reachable,
        duration_s=time.perf_counter() - started,
    )
    validate_artifact(artifact)
    if write:
        write_artifact(artifact, root=base)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run(write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
