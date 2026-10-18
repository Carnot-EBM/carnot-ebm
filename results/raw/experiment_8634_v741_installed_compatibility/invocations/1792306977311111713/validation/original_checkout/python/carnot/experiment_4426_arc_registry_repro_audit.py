"""Exp 4426: ARC solve-registry reproducibility audit.

Spec refs: REQ-REPORT-4426, SCENARIO-REPORT-4426.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4426_arc_registry_repro_audit.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
RANDOM_SEED = 4426
SPEC_REFS = ("REQ-REPORT-4426", "SCENARIO-REPORT-4426")
INFERENCE_SUBSTRATE = "offline_arc_registry_repro_audit_cpu_no_llm"
REQUIRED_ARTIFACT_FIELDS = (
    "reproducible_total_levels",
    "honest_verdict",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "reproducible_total_levels": (
        "Bare int: the single ARC progress metric is recomputed from offline "
        "reproduction evidence instead of trusted from the registry assertion."
    ),
    "honest_verdict": "Terminal-prefixed final audit state.",
    "inference_substrate": "No LLM: CPU/offline ARC simulator plus arc_solver_kit.reproduce.",
}

MILESTONE_409_ARTIFACTS = {
    "exp4421": "results/experiment_4421_config_rule_solve_unseen.json",
    "exp4422": "results/experiment_4422_glyph_rewrite_perception.json",
    "exp4423": "results/experiment_4423_generic_first_contact_breadth.json",
    "exp4424": "results/experiment_4424_deeper_solved_game.json",
}

SCORECARD_PLAN_SOURCES = {
    "lp85": ("results/experiment_4372_e3_deeper_high_headroom_games.json", "per_target_scorecard"),
    "tu93": ("results/experiment_4361_e3_deeper_high_headroom_games.json", "per_target_scorecard"),
    "tn36": ("results/experiment_4372_e3_deeper_high_headroom_games.json", "per_target_scorecard"),
    "ft09": (
        "results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json",
        "per_game_scorecard",
    ),
}

GENERIC_ACTION_ARTIFACTS = {
    "r11l": "results/experiment_4296_arc_incremental_progress_new_game.json",
    "ls20": "results/experiment_4285_arc_incremental_progress_new_game.json",
    "wa30": "results/outer_loop_fable5_wa30_probe_l9.json",
    "cd82": "results/arc_explore_trajectory_cd82.json",
    "sp80": "results/arc_explore_trajectory_sp80.json",
    "su15": "results/arc_explore_trajectory_su15.json",
    "cn04": "results/arc_explore_trajectory_cn04.json",
    "m0r0": "results/arc_explore_trajectory_m0r0.json",
    "sk48": "results/arc_explore_trajectory_sk48.json",
}

ReproduceEntryFn = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]
MetaharnessRunner = Callable[[Path], Mapping[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_registry(root: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {"games": []}
    return loaded if isinstance(loaded, dict) else {"games": []}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _counted_by_registry(entry: Mapping[str, Any]) -> bool:
    return (
        entry.get("reproducibility") == "reproduced" and _as_int(entry.get("levels_reproduced")) > 0
    )


def _terminal_prefixed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(
        ("success:", "complete:", "blocked:", "failed:")
    )


def _actual_reproduced_levels(result: Mapping[str, Any]) -> int:
    return max(0, _as_int(result.get("reached_level", result.get("reproduced_levels"))))


def _entry_audit(
    entry: Mapping[str, Any], reproduction_result: Mapping[str, Any] | None
) -> dict[str, Any]:
    claimed = _as_int(entry.get("levels_reproduced"))
    counted = _counted_by_registry(entry)
    result = dict(reproduction_result or {})
    reached = _actual_reproduced_levels(result) if counted else 0
    claim_reproduced = bool(result.get("reproduced")) and reached >= claimed
    downgraded = bool(counted and not claim_reproduced)
    return {
        "game": str(entry.get("game") or ""),
        "registry_reproducibility": entry.get("reproducibility", ""),
        "registry_levels_reproduced": claimed,
        "counted_by_registry": counted,
        "reproduction_result": result,
        "offline_reproduced_claim": bool(claim_reproduced),
        "effective_levels_reproduced": reached if counted else 0,
        "effective_reproducibility": (
            "provisional"
            if downgraded
            else "reproduced"
            if counted
            else str(entry.get("reproducibility", ""))
        ),
        "downgraded_to_provisional": downgraded,
        "gate": "arc_solver_kit.reproduce" if counted else "not_counted",
    }


def audit_registry(
    root: Path,
    *,
    reproduce_entry_fn: ReproduceEntryFn,
) -> dict[str, Any]:
    registry = load_registry(root)
    rows: list[dict[str, Any]] = []
    for entry in registry.get("games", []):
        if not isinstance(entry, Mapping):
            continue
        result: Mapping[str, Any] | None = None
        if _counted_by_registry(entry):
            try:
                result = reproduce_entry_fn(entry, root)
            except Exception as exc:  # pragma: no cover - defensive live-boundary guard
                result = {
                    "game": entry.get("game", ""),
                    "claimed_level": _as_int(entry.get("levels_reproduced")),
                    "reached_level": 0,
                    "reproduced": False,
                    "mode": f"registry_repro_audit_exception_{type(exc).__name__}",
                    "error": str(exc),
                }
        rows.append(_entry_audit(entry, result))

    downgraded = [row["game"] for row in rows if row["downgraded_to_provisional"]]
    total = sum(_as_int(row["effective_levels_reproduced"]) for row in rows)
    return {
        "registry": registry,
        "rows": rows,
        "downgraded": downgraded,
        "reproducible_total_levels": total,
        "counted_entries_audited": sum(1 for row in rows if row["counted_by_registry"]),
    }


def run_metaharness(
    root: Path = REPO_ROOT,
) -> dict[str, Any]:  # pragma: no cover - subprocess boundary
    python = root / ".venv" / "bin" / "python"
    executable = str(python if python.exists() else Path(sys.executable))
    command = [executable, str(root / "scripts" / "arc3_replay_scorecard_metaharness.py")]
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    path = root / "results" / "arc3_replay_aggregate_scorecard.json"
    payload = _load_json(path)
    per_game = payload.get("per_game") if isinstance(payload.get("per_game"), list) else []
    return {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "artifact_path": str(path.relative_to(root)),
        "total_levels": _as_int(payload.get("total_levels")),
        "games": len(per_game),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _find_scorecard_plan(root: Path, game: str) -> tuple[list[str], str]:  # pragma: no cover
    rel_path, row_key = SCORECARD_PLAN_SOURCES[game]
    artifact = _load_json(root / rel_path)
    rows = artifact.get(row_key)
    if isinstance(rows, list):
        for row in rows:
            if (
                isinstance(row, Mapping)
                and row.get("game") == game
                and isinstance(row.get("plan"), list)
            ):
                return [str(label) for label in row["plan"]], rel_path
    return [], rel_path


def _normalize_action(action: Mapping[str, Any]) -> tuple[int | None, dict[str, int] | None]:
    action_id = action.get("action")
    if "data" in action:
        return (_as_int(action_id) if action_id is not None else None), action.get("data")
    x = action.get("x", action.get("world_x"))
    y = action.get("y", action.get("world_y"))
    if x is None or y is None:
        return (_as_int(action_id) if action_id is not None else None), None
    return (_as_int(action_id) if action_id is not None else 6), {"x": _as_int(x), "y": _as_int(y)}


def _load_action_dicts(root: Path, rel_path: str) -> list[Mapping[str, Any]]:  # pragma: no cover
    artifact = _load_json(root / rel_path)
    for keys in (
        ("solution",),
        ("trajectory",),
        ("action_sequence",),
        ("solve_trace", "actions"),
        ("solver_trace", "actions"),
        ("plan_executed_detail", "plan_result", "executed_steps"),
    ):
        current: Any = artifact
        for key in keys:
            current = current.get(key) if isinstance(current, Mapping) else None
            if current is None:
                break
        if isinstance(current, list) and current:
            return [row for row in current if isinstance(row, Mapping)]
    return []


def _labels_from_action_artifact(
    root: Path, game: str
) -> tuple[list[str], str]:  # pragma: no cover
    rel_path = GENERIC_ACTION_ARTIFACTS[game]
    labels = []
    for action in _load_action_dicts(root, rel_path):
        action_id, data = _normalize_action(action)
        if action_id is not None:
            payload: dict[str, Any] = {"action": action_id}
            if data is not None:
                payload["data"] = data
            labels.append(json.dumps(payload, sort_keys=True))
    return labels, rel_path


def _labels_for_game(root: Path, game: str) -> tuple[list[str], str]:  # pragma: no cover
    if game in SCORECARD_PLAN_SOURCES:
        return _find_scorecard_plan(root, game)
    if game == "tr87":
        rel_path = "results/arc_loop_solve_tr87.json"
        artifact = _load_json(root / rel_path)
        labels = artifact.get("solution_labels") or []
        return [str(label) for label in labels], rel_path
    if game in GENERIC_ACTION_ARTIFACTS:
        return _labels_from_action_artifact(root, game)
    return [], ""


def _generic_apply_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    parsed: Any = json.loads(label)
    if isinstance(parsed, int):
        action_id, data = parsed, None
    elif isinstance(parsed, Mapping):
        action_id = parsed.get("action")
        data = parsed.get("data")
        if action_id is None and parsed.get("x") is not None and parsed.get("y") is not None:
            action_id = 6
            data = {"x": _as_int(parsed["x"]), "y": _as_int(parsed["y"])}
    else:
        raise ValueError(f"unsupported action label: {label!r}")
    return env.step(_game_action(GameAction, _as_int(action_id)), data=data)


def _kit_reproduce(
    game: str,
    labels: Sequence[str],
    apply_fn: Callable[[Any, str, Any], Any],
    *,
    claimed_level: int,
    source: str,
    warmup_label: str | None = None,
) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit

    result = dict(
        arc_solver_kit.reproduce(
            game,
            labels,
            apply_fn,
            warmup_label=warmup_label,
            claimed_level=claimed_level,
        )
    )
    result["source"] = source
    result["gate"] = "arc_solver_kit.reproduce"
    return result


def reproduce_registry_entry(
    entry: Mapping[str, Any], root: Path = REPO_ROOT
) -> dict[str, Any]:  # pragma: no cover
    """REQ-REPORT-4426: replay one counted registry row through arc_solver_kit.reproduce."""

    game = str(entry.get("game") or "")
    claimed = _as_int(entry.get("levels_reproduced"))
    if game == "s5i5":  # pragma: no cover - ARC SDK boundary
        from carnot import experiment_4421_config_rule_solve_unseen as exp4421

        artifact = _load_json(root / exp4421.RESULT_RELATIVE_PATH)
        solution = artifact.get("solver", {}).get("solution") or exp4421.derive_s5i5_l1_path()
        return _kit_reproduce(
            game,
            [str(label) for label in solution],
            exp4421.apply_s5i5_label,
            claimed_level=claimed,
            source=exp4421.RESULT_RELATIVE_PATH,
        )
    if game == "sc25":  # pragma: no cover - ARC SDK boundary
        from carnot.experiment_4341_e3_sc25_reproduction import (
            L1_SOLUTION_LABELS,
            _apply_sc25_label,
        )

        return _kit_reproduce(
            game,
            L1_SOLUTION_LABELS,
            _apply_sc25_label,
            warmup_label="warmup",
            claimed_level=claimed,
            source="python/carnot/experiment_4341_e3_sc25_reproduction.py",
        )
    if game == "ar25":  # pragma: no cover - ARC SDK boundary
        from carnot.experiment_4339_e3_explore_verify_plan_ar25 import (
            L1_SOLUTION_LABELS,
            _apply_ar25_label,
        )

        return _kit_reproduce(
            game,
            L1_SOLUTION_LABELS,
            _apply_ar25_label,
            claimed_level=claimed,
            source="python/carnot/experiment_4339_e3_explore_verify_plan_ar25.py",
        )
    if game == "ka59":  # pragma: no cover - ARC SDK boundary
        from carnot.experiment_4350_e3_explore_verify_plan_ka59 import (
            L1_SOLUTION_LABELS,
            _apply_ka59_label,
        )

        return _kit_reproduce(
            game,
            L1_SOLUTION_LABELS,
            _apply_ka59_label,
            claimed_level=claimed,
            source="python/carnot/experiment_4350_e3_explore_verify_plan_ka59.py",
        )

    labels, source = _labels_for_game(root, game)  # pragma: no cover - ARC SDK boundary
    if not labels:  # pragma: no cover - ARC SDK boundary
        return {
            "game": game,
            "claimed_level": claimed,
            "reached_level": 0,
            "reproduced": False,
            "mode": "missing_replay_plan_for_registry_entry",
            "source": source,
        }
    return _kit_reproduce(game, labels, _generic_apply_label, claimed_level=claimed, source=source)


def _gate_evidence(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "reproduction_result",
        "reproduce_result",
        "standing_loop_result",
        "reproduction_gate",
    ):
        value = artifact.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def milestone_409_gate_rows(root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp_id, rel_path in MILESTONE_409_ARTIFACTS.items():
        artifact = _load_json(root / rel_path)
        evidence = _gate_evidence(artifact)
        evidence_present = bool(evidence)
        offline_reproduced = bool(artifact.get("offline_reproduced"))
        reproduced_levels = _as_int(artifact.get("reproduced_levels"))
        explicit_new = artifact.get("new_levels_reproduced")
        new_levels = _as_int(explicit_new) if explicit_new is not None and offline_reproduced else 0
        rows.append(
            {
                "experiment": exp_id,
                "artifact": rel_path,
                "artifact_present": bool(artifact),
                "reproduction_gated": evidence_present,
                "offline_reproduced": offline_reproduced,
                "reproduced_levels": reproduced_levels,
                "new_levels_counted": new_levels,
                "artifact_flagged_adversarial": bool(artifact.get("flagged_adversarial")),
                "honest_verdict": artifact.get("honest_verdict", ""),
                "gate_evidence_keys": sorted(str(key) for key in evidence.keys()),
            }
        )
    return rows


def _honest_verdict(total: int, registry_claimed: int, downgraded: Sequence[str]) -> str:
    if downgraded:
        return f"complete: registry_repro_audit_flagged_{len(downgraded)}_counted_entries"
    if total != registry_claimed:
        return f"complete: registry_repro_audit_total_{registry_claimed}_asserted_{total}_audited"
    return f"success: registry_reproducible_total_levels_{total}_audited"


def build_artifact(
    *,
    registry_audit: Mapping[str, Any],
    metaharness: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    started_at: float,
    ended_at: float,
) -> dict[str, Any]:
    registry = (
        registry_audit.get("registry")
        if isinstance(registry_audit.get("registry"), Mapping)
        else {}
    )
    rows = registry_audit.get("rows") if isinstance(registry_audit.get("rows"), list) else []
    downgraded = (
        registry_audit.get("downgraded")
        if isinstance(registry_audit.get("downgraded"), list)
        else []
    )
    total = _as_int(registry_audit.get("reproducible_total_levels"))
    claimed_total = _as_int(registry.get("reproducible_total_levels"))
    artifact = {
        "experiment": "experiment_4426_arc_registry_repro_audit",
        "schema": "carnot.exp4426.arc_registry_repro_audit.v1",
        "reproducible_total_levels": total,
        "registry_claimed_reproducible_total_levels": claimed_total,
        "registry_claimed_reproducible_total_games": _as_int(
            registry.get("reproducible_total_games")
        ),
        "counted_entries_audited": _as_int(registry_audit.get("counted_entries_audited")),
        "registry_entry_audits": [dict(row) for row in rows],
        "entries_downgraded_to_provisional": [str(game) for game in downgraded],
        "all_counted_entries_reproduced": not downgraded,
        "metaharness": dict(metaharness),
        "milestone_409_reproduction_gates": [dict(row) for row in gate_rows],
        "honest_verdict": _honest_verdict(total, claimed_total, [str(game) for game in downgraded]),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "submitted_to_leaderboard": False,
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": max(0.0, round(float(ended_at - started_at), 6)),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if type(artifact.get("reproducible_total_levels")) is not int:
        errors.append("reproducible_total_levels must be bare int")
    if not _terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("inference_substrate"), str) or not artifact.get(
        "inference_substrate"
    ):
        errors.append("inference_substrate must be non-empty string")
    if not isinstance(artifact.get("registry_entry_audits"), list):
        errors.append("registry_entry_audits must be list")
    if not isinstance(artifact.get("milestone_409_reproduction_gates"), list):
        errors.append("milestone_409_reproduction_gates must be list")
    if not isinstance(artifact.get("metaharness"), Mapping):
        errors.append("metaharness must be dict")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    reproduce_entry_fn: ReproduceEntryFn = reproduce_registry_entry,
    metaharness_runner: MetaharnessRunner = run_metaharness,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """SCENARIO-REPORT-4426: run the replay harness, audit registry rows, and write JSON."""

    started = now()
    metaharness = dict(metaharness_runner(root))
    registry_audit = audit_registry(root, reproduce_entry_fn=reproduce_entry_fn)
    artifact = build_artifact(
        registry_audit=registry_audit,
        metaharness=metaharness,
        gate_rows=milestone_409_gate_rows(root),
        started_at=started,
        ended_at=now(),
    )
    write_artifact(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
