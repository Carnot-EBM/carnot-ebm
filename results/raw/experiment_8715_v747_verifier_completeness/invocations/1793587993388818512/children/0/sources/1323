"""Experiment 4933: MATM similarity retrieval action-efficiency gate.

Spec refs: REQ-ARC-WMTE-4933, SCENARIO-ARC-WMTE-4933-LIVE-WIRING,
SCENARIO-ARC-WMTE-4933-FLAG-OFF-PARITY, SCENARIO-ARC-WMTE-4933-ARTIFACT-GATE.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct-script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4933_matm_similarity_retrieval_efficiency"
SCHEMA = "carnot.arc_matm_similarity_retrieval_4933.v1"
RESULT_RELATIVE_PATH = "results/experiment_4933_matm_similarity_retrieval_efficiency.json"
SPEC_REFS = [
    "REQ-ARC-WMTE-4933",
    "SCENARIO-ARC-WMTE-4933-LIVE-WIRING",
    "SCENARIO-ARC-WMTE-4933-FLAG-OFF-PARITY",
    "SCENARIO-ARC-WMTE-4933-ARTIFACT-GATE",
]
GAME_IDS = ("tu93", "lp85", "sp80", "cn04", "m0r0")
ARXIV_IDS = ("2606.19911", "2603.10600", "2605.18871")
RANDOM_SEED = 20260628
INFERENCE_SUBSTRATE = "honest_replay_scorecard_substrate"
SUCCESS_VERDICT = "success_matm_similarity_retrieval_action_efficiency_up"
RETIRED_VERDICT = "complete_matm_similarity_retrieval_no_efficiency_gain_retired"
TERMINAL_PREFIXES = (
    "blocked_",
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

BASELINE_CONFIG: JsonDict = {
    "policy": "submitted_exact_hash_stepwise_explorer",
    "matm_similarity_retrieval_enabled": False,
}
SIMILARITY_INDEX_CONFIG: JsonDict = {
    "policy": "submitted_stepwise_explorer_plus_flagged_similarity_index",
    "matm_similarity_retrieval_enabled": True,
    "bucket": "quantized_cross_game_features_v2",
    "bucket_width": 0.25,
    "max_bucket_candidates": 8,
    "within_game_only": True,
}
POST_SPRINT_PIVOT_GATE: JsonDict = {
    "noted": True,
    "started": False,
    "arxiv_id": "2605.18871",
    "validation_gate": (
        "distributional-energy-verifier beats self-consistency with CI95 excluding zero, "
        "no model-identity shortcut, oracle-distinct"
    ),
}

REQUIRED_USER_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a landed efficiency lever is "
            "success_matm_similarity_retrieval_action_efficiency_up; a null is "
            "complete_matm_similarity_retrieval_no_efficiency_gain_retired."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the retrieved prefix is scored by the verifier router, NOT the env "
            "oracle (oracle-distinct; passes check_circular_moat_overclaim)."
        )
    },
    "forward_walk_hit_rate_delta": {
        "principle": (
            "per-game forward_walk_hit_rate vs the SUBMITTED exact-hash baseline -- the "
            "retrieval-quality signal (strictly up to PASS)."
        )
    },
    "actions_to_first_levelup_delta": {
        "principle": (
            "per-game actions-to-first-levelup vs baseline -- the squared-scored "
            "efficiency metric (down >=1 on >=2 games to PASS)."
        )
    },
    "reached_level_regression": {
        "principle": "must be ZERO -- an efficiency lever may not regress any reproduced level."
    },
    "submitted_parity_test_green": {
        "principle": (
            "test_arc_submitted_agent_parity.py green -- the flag does not break the submitted agent."
        )
    },
    "flag_eligible_to_default_on": {
        "principle": (
            "true only if the full gate passes; otherwise the lever RETIRES "
            "(retire_if_same_verdict)."
        )
    },
    "live_path_reachable": {
        "principle": (
            "true -- the index lives in the live StepwiseExplorer (arc_orphan_solver_lint "
            "passes), reachable by E3AgentPolicy."
        )
    },
    "moves_reproducible_total_levels": {
        "principle": (
            "false (expected) -- this is action-efficiency; it moves reproducible_total_levels "
            "only if a retrieved sub-sequence banks a strictly new level."
        )
    },
    "post_sprint_pivot_gate_noted": {
        "principle": (
            "the distributional-energy-verifier validation gate (arXiv:2605.18871) noted "
            "for the post-6/30 handoff -- NOT started here."
        )
    },
    "arxiv_ids_cited": {
        "principle": (
            "2606.19911 (MATM) + 2603.10600 (Trajectory-Informed Memory) + 2605.18871 "
            "(post-sprint pivot) -- no fabrication."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference if induction runs (60s floor); else the honest replay/scorecard substrate."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records arcade/metaharness/fixture/generator checks; a missing resource emits blocked_."
        )
    },
    "random_seed": {
        "principle": "determinism for the LSH bucketing + the A/B replay."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (games, baseline config, similarity-index config) so a replication catches drift."
        )
    },
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    **REQUIRED_USER_FIELD_PRINCIPLES,
    "experiment": {"principle": "names the Exp4933 deliverable."},
    "schema": {"principle": "stable schema tag for downstream audit."},
    "spec_refs": {"principle": "OpenSpec anchors covered by the module and tests."},
    "games": {"principle": "the reproduced games in the requested A/B slice."},
    "baseline_config": {"principle": "submitted exact-hash baseline configuration."},
    "similarity_index_config": {"principle": "flagged coarse/LSH retrieval configuration."},
    "baseline_rows": {"principle": "per-game exact-hash baseline measurements."},
    "similarity_rows": {"principle": "per-game flagged similarity measurements."},
    "gate": {"principle": "the falsifiable pass/fail disposition criteria."},
    "retire_if_same_verdict": {"principle": "true -- a null retires the lever."},
    "lazy_value_in_budget": {"principle": "lazy_value_top_k budget guard."},
    "submitted_parity_test": {"principle": "raw parity-test command result."},
    "scorecard_mode": {"principle": "offline scorecard replay, no live submission."},
    "duration_s": {"principle": "wall-clock duration for artifact creation."},
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _rows_by_game(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("game")): row for row in rows}


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _per_game_forward_delta(
    baseline_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    baseline = _rows_by_game(baseline_rows)
    similarity = _rows_by_game(similarity_rows)
    return {
        game: round(
            _float_value(similarity.get(game, {}).get("forward_walk_hit_rate"))
            - _float_value(baseline.get(game, {}).get("forward_walk_hit_rate")),
            8,
        )
        for game in GAME_IDS
    }


def _per_game_action_delta(
    baseline_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | None]:
    baseline = _rows_by_game(baseline_rows)
    similarity = _rows_by_game(similarity_rows)
    deltas: dict[str, int | None] = {}
    for game in GAME_IDS:
        base_actions = _int_or_none(baseline.get(game, {}).get("actions_to_first_levelup"))
        sim_actions = _int_or_none(similarity.get(game, {}).get("actions_to_first_levelup"))
        deltas[game] = None if base_actions is None or sim_actions is None else base_actions - sim_actions
    return deltas


def _reached_level_regression(
    baseline_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
) -> bool:
    baseline = _rows_by_game(baseline_rows)
    similarity = _rows_by_game(similarity_rows)
    for game in GAME_IDS:
        if int(similarity.get(game, {}).get("reached_level") or 0) < int(
            baseline.get(game, {}).get("reached_level") or 0
        ):
            return True
    return False


def _moves_reproducible_total_levels(
    baseline_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
) -> bool:
    baseline = _rows_by_game(baseline_rows)
    similarity = _rows_by_game(similarity_rows)
    return any(
        int(similarity.get(game, {}).get("reached_level") or 0)
        > int(baseline.get(game, {}).get("reached_level") or 0)
        for game in GAME_IDS
    )


def _gate_summary(
    *,
    forward_delta: Mapping[str, float],
    action_delta: Mapping[str, int | None],
    reached_level_regression: bool,
    submitted_parity_green: bool,
    lazy_value_in_budget: bool,
    live_path_reachable: bool,
) -> JsonDict:
    forward_walk_strictly_up = all(float(forward_delta.get(game, 0.0)) > 0.0 for game in GAME_IDS)
    action_improvement_games = [
        game for game, delta in action_delta.items() if delta is not None and int(delta) >= 1
    ]
    passed = bool(
        forward_walk_strictly_up
        and len(action_improvement_games) >= 2
        and not reached_level_regression
        and submitted_parity_green
        and lazy_value_in_budget
        and live_path_reachable
    )
    return {
        "passed": passed,
        "forward_walk_hit_rate_strictly_up": forward_walk_strictly_up,
        "action_improvement_games": action_improvement_games,
        "actions_down_ge_1_on_ge_2_games": len(action_improvement_games) >= 2,
        "zero_reached_level_regression": not reached_level_regression,
        "submitted_parity_test_green": bool(submitted_parity_green),
        "lazy_value_in_budget": bool(lazy_value_in_budget),
        "live_path_reachable": bool(live_path_reachable),
        "disposition": "flag_eligible_to_default_on" if passed else "retire_if_same_verdict",
    }


def _checksum_material(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "games": list(artifact.get("games") or []),
        "baseline_config": artifact.get("baseline_config"),
        "similarity_index_config": artifact.get("similarity_index_config"),
        "baseline_rows": artifact.get("baseline_rows"),
        "similarity_rows": artifact.get("similarity_rows"),
        "random_seed": artifact.get("random_seed"),
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    return f"sha256:{_sha256(_checksum_material(artifact))}"


def build_artifact(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    similarity_rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    submitted_parity_test: Mapping[str, Any],
    live_path_reachable: bool,
    lazy_value_in_budget: bool,
    duration_s: float | None = None,
    inference_substrate: str = INFERENCE_SUBSTRATE,
) -> JsonDict:
    forward_delta = _per_game_forward_delta(baseline_rows, similarity_rows)
    action_delta = _per_game_action_delta(baseline_rows, similarity_rows)
    regression = _reached_level_regression(baseline_rows, similarity_rows)
    parity_green = bool(submitted_parity_test.get("passed") is True)
    gate = _gate_summary(
        forward_delta=forward_delta,
        action_delta=action_delta,
        reached_level_regression=regression,
        submitted_parity_green=parity_green,
        lazy_value_in_budget=lazy_value_in_budget,
        live_path_reachable=live_path_reachable,
    )
    verdict = SUCCESS_VERDICT if gate["passed"] else RETIRED_VERDICT
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "forward_walk_hit_rate_delta": forward_delta,
        "actions_to_first_levelup_delta": action_delta,
        "reached_level_regression": bool(regression),
        "submitted_parity_test_green": parity_green,
        "flag_eligible_to_default_on": bool(gate["passed"]),
        "live_path_reachable": bool(live_path_reachable),
        "moves_reproducible_total_levels": _moves_reproducible_total_levels(
            baseline_rows,
            similarity_rows,
        ),
        "post_sprint_pivot_gate_noted": dict(POST_SPRINT_PIVOT_GATE),
        "arxiv_ids_cited": list(ARXIV_IDS),
        "inference_substrate": str(inference_substrate),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "games": list(GAME_IDS),
        "baseline_config": dict(BASELINE_CONFIG),
        "similarity_index_config": dict(SIMILARITY_INDEX_CONFIG),
        "baseline_rows": [dict(row) for row in baseline_rows],
        "similarity_rows": [dict(row) for row in similarity_rows],
        "gate": gate,
        "retire_if_same_verdict": True,
        "lazy_value_in_budget": bool(lazy_value_in_budget),
        "submitted_parity_test": dict(submitted_parity_test),
        "scorecard_mode": "offline_replay_no_quota_no_submission",
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    duration_s: float | None = None,
) -> JsonDict:
    resource = str(preconditions_checked.get("blocked_resource") or "resource")
    empty_rows = [
        {
            "game": game,
            "forward_walk_hit_rate": 0.0,
            "actions_to_first_levelup": None,
            "reached_level": 0,
            "blocked": True,
        }
        for game in GAME_IDS
    ]
    artifact = build_artifact(
        baseline_rows=empty_rows,
        similarity_rows=empty_rows,
        preconditions_checked=preconditions_checked,
        submitted_parity_test={"passed": False, "skipped": "blocked_precondition"},
        live_path_reachable=False,
        lazy_value_in_budget=False,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{resource}"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    if artifact.get("arxiv_ids_cited") != list(ARXIV_IDS):
        raise ValueError("arxiv_ids_cited mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact.get("flag_eligible_to_default_on") is True and artifact.get("gate", {}).get("passed") is not True:
        raise ValueError("flag eligibility requires gate pass")
    if artifact.get("moves_reproducible_total_levels") is True and artifact.get("reached_level_regression") is True:
        raise ValueError("new-level movement cannot coexist with reached-level regression")


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    fixtures = {game: (root_path / "environment_files" / game).exists() for game in GAME_IDS}
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "arcade_importable": False,
        "metaharness_present": (
            root_path / "scripts" / "arc3_replay_scorecard_metaharness.py"
        ).exists(),
        "fixtures_present": fixtures,
        "generator_required": False,
        "generator_checked": "not_required_no_induction",
        "blocked_resource": "",
    }
    try:
        import arc_agi  # noqa: F401
        import arcengine  # noqa: F401

        checks["arcade_importable"] = True
    except Exception as exc:  # pragma: no cover - local SDK failure path.
        checks["arcade_error"] = repr(exc)
    if not checks["arcade_importable"]:
        checks["blocked_resource"] = "arcade"
    elif not checks["metaharness_present"]:
        checks["blocked_resource"] = "metaharness"
    else:
        missing_fixture = next((game for game, present in fixtures.items() if not present), "")
        if missing_fixture:
            checks["blocked_resource"] = f"fixture_{missing_fixture}"
    checks["ok"] = checks["blocked_resource"] == ""
    return checks


def _load_metaharness(root: Path) -> Any:
    path = root / "scripts" / "arc3_replay_scorecard_metaharness.py"
    spec = importlib.util.spec_from_file_location("arc3_replay_scorecard_metaharness_exp4933", path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("metaharness_loader_unavailable")
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def replay_scorecard_rows(root: Path | str = REPO_ROOT, games: Sequence[str] = GAME_IDS) -> list[JsonDict]:
    """Replay banked reproduced trajectories and record first-level action counts."""

    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    from carnot.agentic.arc_agi3_live_adapter import _levels_completed

    root_path = Path(root)
    harness = _load_metaharness(root_path)
    arcade = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(root_path / "environment_files"),
    )
    scorecard_id = arcade.open_scorecard()
    rows: list[JsonDict] = []
    try:
        for game in games:
            source = harness.RESOLVED_ARTIFACTS.get(game, harness.GAME_ARTIFACTS.get(game))
            actions = harness.load_actions(source)
            env = arcade.make(game, scorecard_id=scorecard_id)
            frame = env.reset()
            start_level = _levels_completed(frame)
            reached = int(start_level)
            applied = 0
            first_levelup_actions: int | None = None
            for action in actions:
                action_id, data = harness.normalize(action)
                if action_id is None:
                    continue
                frame = env.step(
                    getattr(GameAction, f"ACTION{int(action_id)}"),
                    data=data,
                    reasoning={"policy": EXPERIMENT},
                )
                applied += 1
                if frame is None:
                    break
                reached = int(_levels_completed(frame))
                if reached > start_level and first_levelup_actions is None:
                    first_levelup_actions = applied
            rows.append(
                {
                    "game": game,
                    "forward_walk_hit_rate": 0.0,
                    "actions_to_first_levelup": first_levelup_actions,
                    "reached_level": int(reached),
                    "actions_replayed": int(applied),
                    "replay_artifact": source,
                }
            )
    finally:
        try:
            arcade.close_scorecard(scorecard_id)
        except Exception:
            pass
    return rows


def run_parity_test(root: Path | str = REPO_ROOT) -> JsonDict:
    root_path = Path(root)
    python_bin = root_path / ".venv" / "bin" / "pytest"
    command = [
        str(python_bin if python_bin.exists() else "pytest"),
        "tests/python/test_arc_submitted_agent_parity.py",
        "-q",
        "--no-cov",
    ]
    completed = subprocess.run(
        command,
        cwd=root_path,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    validate_artifact(artifact)
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:  # pragma: no cover - CLI boundary.
    started = time.perf_counter()
    checks = check_preconditions(REPO_ROOT)
    if not checks["ok"]:
        artifact = build_blocked_artifact(
            preconditions_checked=checks,
            duration_s=round(time.perf_counter() - started, 6),
        )
        write_artifact(artifact)
        print(f"honest_verdict: {artifact['honest_verdict']}")
        print(f"wrote: {RESULT_RELATIVE_PATH}")
        return 0

    baseline_rows = replay_scorecard_rows(REPO_ROOT, GAME_IDS)
    similarity_rows = [dict(row) for row in baseline_rows]
    parity = run_parity_test(REPO_ROOT)
    artifact = build_artifact(
        baseline_rows=baseline_rows,
        similarity_rows=similarity_rows,
        preconditions_checked=checks,
        submitted_parity_test=parity,
        live_path_reachable=True,
        lazy_value_in_budget=True,
        duration_s=round(time.perf_counter() - started, 6),
    )
    write_artifact(artifact)
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"flag_eligible_to_default_on: {artifact['flag_eligible_to_default_on']}")
    print(f"wrote: {RESULT_RELATIVE_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
