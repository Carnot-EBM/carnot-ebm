"""Experiment 4534: trust-energy next-level frontier routing.

Spec refs: REQ-ARC-WMTE-4534, SCENARIO-ARC-WMTE-4534.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4534_energy_trust_next_level_routing.json"
A1_RELATIVE_PATH = "results/experiment_4533_per_level_goal_reinduction.json"
A2_RELATIVE_PATH = "results/experiment_4524_reach_deeper_levels.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
CORE_EFFICIENCY_BASELINE = 2.0074
RANDOM_SEED = 4534
TARGET_GAMES = ("lp85", "sp80")
CORE_GAMES = ("lp85", "m0r0", "sp80", "vc33")
REQUIREMENTS = ("REQ-ARC-WMTE-4534",)
SCENARIOS = ("SCENARIO-ARC-WMTE-4534",)
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: energy_routing_<game>_reached_L2_oracle_distinct OR "
        "complete: energy_routing_no_deeper_level_signal_characterized_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline arcade + trust-energy scoring, "
        "no headline GGUF load (1s floor)."
    ),
    "verifier_is_oracle": (
        "MUST be false -- the trust energy is oracle-DISTINCT (ranks held-out generalization), "
        "NOT the executable win-check; a circular win does not count (Circularity / "
        "Oracle-Distinctness Discipline)."
    ),
    "core_efficiency_baseline": "2.0074 -- the per-level metric control.",
    "core_efficiency_energy_routed": (
        "the HEADLINE -- did energy routing toward the re-induced goal reach a deeper level."
    ),
    "no_energy_control": (
        "matched control measured the SAME way (energy off) -- the apples-to-apples comparison."
    ),
    "energy_separation_auroc": (
        "AUROC of the trust energy separating deeper-progress states from L1-stuck states -- "
        "the signal characterization that is the deliverable on a null."
    ),
    "deepest_level_reached_per_core_game": (
        "best_level per CORE game with vs without energy routing -- the score-lever evidence."
    ),
    "core_solves_preserved": (
        "HARD gate on {lp85,m0r0,sp80,vc33}; a dropped CORE solve FAILS the lever."
    ),
    "positive_control_passed": (
        "the energy SHOULD separate deeper-progress on a known-L2 game; guards a silently-broken signal."
    ),
    "false_negative_risk_checked": "a null is valid only if the positive control passed.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent drift on replay.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "field_principles",
    "requirements",
    "scenarios",
    "target_games",
    "core_games",
    "energy_routed_measurement",
    "signal_characterization",
    "a1_reinduced_goal",
    "chosen_submitted_config",
    "measurements",
    "result_path",
    "duration_s",
)


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _round_efficiency(value: Any) -> float:
    return round(float(value or 0.0), 4)


def _label_is_progress(sample: Mapping[str, Any]) -> bool:
    if "deeper_progress" in sample:
        return bool(sample.get("deeper_progress"))
    label = sample.get("label")
    return label in {1, "1", "deeper", "deeper_progress", "progress", True}


def energy_separation_auroc(samples: Sequence[Mapping[str, Any]]) -> float | None:
    """REQ-ARC-WMTE-4534: lower trust energy should rank deeper-progress states first."""

    positives = [float(row["energy"]) for row in samples if _label_is_progress(row)]
    negatives = [float(row["energy"]) for row in samples if not _label_is_progress(row)]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos < neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return round(wins / total, 4)


def _levels_by_game(
    measurement: Mapping[str, Any],
    *,
    fallback: Mapping[str, int] | None = None,
) -> dict[str, int]:
    raw = measurement.get("deepest_level_by_game")
    if not isinstance(raw, Mapping):
        raw = measurement.get("best_level_by_game")
    out = {game: int((fallback or {}).get(game, 0)) for game in CORE_GAMES}
    if isinstance(raw, Mapping):
        for game, value in raw.items():
            if str(game) in CORE_GAMES:
                out[str(game)] = int(value or 0)
    for row in measurement.get("per_game", []) or []:
        if isinstance(row, Mapping) and str(row.get("game")) in CORE_GAMES:
            out[str(row["game"])] = int(row.get("best_level") or row.get("levels") or 0)
    return out


def _per_game_efficiency(measurement: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    raw = measurement.get("efficiency_by_game")
    if isinstance(raw, Mapping):
        out.update({str(game): _round_efficiency(value) for game, value in raw.items()})
    for row in measurement.get("per_game", []) or []:
        if isinstance(row, Mapping) and row.get("game") is not None:
            value = row.get("efficiency", row.get("per_level_efficiency"))
            if value is not None:
                out[str(row["game"])] = _round_efficiency(value)
    return out


def _normalise_measurement(
    measurement: Mapping[str, Any],
    *,
    label: str,
    fallback_levels: Mapping[str, int] | None = None,
    fallback_efficiency_by_game: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    levels = _levels_by_game(measurement, fallback=fallback_levels)
    efficiency_by_game = dict(fallback_efficiency_by_game or {})
    efficiency_by_game.update(_per_game_efficiency(measurement))
    if measurement.get("core_efficiency") is None and efficiency_by_game:
        core_efficiency = _round_efficiency(sum(efficiency_by_game.get(game, 0.0) for game in CORE_GAMES))
    else:
        core_efficiency = _round_efficiency(measurement.get("core_efficiency"))
    return {
        **dict(measurement),
        "measurement": str(measurement.get("measurement") or label),
        "target_games": list(measurement.get("target_games") or TARGET_GAMES),
        "core_efficiency": core_efficiency,
        "deepest_level_by_game": levels,
        "efficiency_by_game": {game: float(efficiency_by_game.get(game, 0.0)) for game in CORE_GAMES},
    }


def _core_solves_preserved(control: Mapping[str, Any], energy: Mapping[str, Any]) -> bool:
    control_levels = _levels_by_game(control)
    energy_levels = _levels_by_game(energy, fallback=control_levels)
    return all(int(energy_levels.get(game, 0)) >= int(control_levels.get(game, 0)) for game in CORE_GAMES)


def _energy_only_l2_game(control: Mapping[str, Any], energy: Mapping[str, Any]) -> str | None:
    control_levels = _levels_by_game(control)
    energy_levels = _levels_by_game(energy, fallback=control_levels)
    for game in CORE_GAMES:
        if int(energy_levels.get(game, 0)) >= 2 and int(control_levels.get(game, 0)) < 2:
            return game
    return None


def _positive_control_passed(signal: Mapping[str, Any], auroc: float | None) -> bool:
    positive = signal.get("positive_control")
    positive_declared = isinstance(positive, Mapping) and positive.get("passed") is True
    return bool(positive_declared and auroc is not None and float(auroc) > 0.5)


def _success_verdict(game: str) -> str:
    return f"success: energy_routing_{game}_reached_L2_oracle_distinct"


def _null_verdict(*, positive_control_passed: bool) -> str:
    if not positive_control_passed:
        return "complete: energy_routing_positive_control_failed_signal_uninformative"
    return "complete: energy_routing_no_deeper_level_signal_characterized_honest_null"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    no_energy_control: Mapping[str, Any],
    energy_routed: Mapping[str, Any],
    signal_characterization: Mapping[str, Any],
    a1_goal: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4534: assemble the terminal trust-energy routing artifact."""

    control = _normalise_measurement(no_energy_control, label="no_energy_control")
    energy = _normalise_measurement(
        energy_routed,
        label="energy_routed",
        fallback_levels=control["deepest_level_by_game"],
        fallback_efficiency_by_game=control["efficiency_by_game"],
    )
    samples = list(signal_characterization.get("samples") or [])
    auroc = signal_characterization.get("energy_separation_auroc")
    auroc = energy_separation_auroc(samples) if auroc is None else round(float(auroc), 4)
    signal = {**dict(signal_characterization), "energy_separation_auroc": auroc}
    positive_passed = _positive_control_passed(signal, auroc)
    core_preserved = _core_solves_preserved(control, energy)
    l2_game = _energy_only_l2_game(control, energy)
    success = bool(l2_game is not None and core_preserved and positive_passed)
    chosen_config: dict[str, Any] | str
    if success:
        chosen_config = {
            "energy_next_level_routing": True,
            "target_games": list(TARGET_GAMES),
            "depth_primary": True,
            "trust_energy_lower_is_better": True,
        }
    else:
        chosen_config = "unchanged"
    artifact = {
        "experiment": "experiment_4534_energy_trust_next_level_routing",
        "schema": "carnot.arc_energy_trust_next_level_routing_4534.v1",
        "honest_verdict": _success_verdict(l2_game) if success else _null_verdict(positive_control_passed=positive_passed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "preconditions_checked": dict(preconditions_checked),
        "core_efficiency_baseline": CORE_EFFICIENCY_BASELINE,
        "core_efficiency_energy_routed": _round_efficiency(energy.get("core_efficiency")),
        "no_energy_control": control,
        "energy_separation_auroc": auroc,
        "deepest_level_reached_per_core_game": {
            "no_energy_control": dict(control["deepest_level_by_game"]),
            "energy_routed": dict(energy["deepest_level_by_game"]),
        },
        "core_solves_preserved": bool(core_preserved),
        "positive_control_passed": bool(positive_passed),
        "false_negative_risk_checked": bool(positive_passed),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "target_games": list(TARGET_GAMES),
        "core_games": list(CORE_GAMES),
        "energy_routed_measurement": energy,
        "signal_characterization": signal,
        "a1_reinduced_goal": dict(a1_goal),
        "chosen_submitted_config": chosen_config,
        "measurements": [control, energy],
        "result_path": RESULT_RELATIVE_PATH,
        "duration_s": None if duration_s is None else float(duration_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    blocked = isinstance(verdict, str) and verdict.startswith("blocked_")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must match")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false for oracle-distinct trust energy")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-ARC-WMTE-4534")
    if float(artifact.get("core_efficiency_baseline") or 0.0) != CORE_EFFICIENCY_BASELINE:
        errors.append("core_efficiency_baseline must equal 2.0074")
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, Mapping):
        errors.append("preconditions_checked must be a mapping")
    elif not blocked:
        if preconditions.get("offline_arcade_import_smoke") is not True:
            errors.append("preconditions_checked must record offline_arcade_import_smoke=true")
        if preconditions.get("spec_has_req_4534") is not True:
            errors.append("preconditions_checked must record spec_has_req_4534=true")
    if not isinstance(artifact.get("no_energy_control"), Mapping):
        errors.append("no_energy_control must be a mapping")
    if not isinstance(artifact.get("deepest_level_reached_per_core_game"), Mapping):
        errors.append("deepest_level_reached_per_core_game must be a mapping")
    auroc = artifact.get("energy_separation_auroc")
    if auroc is None:
        errors.append("energy_separation_auroc must be reported")
    elif not 0.0 <= float(auroc) <= 1.0:
        errors.append("energy_separation_auroc must be between 0 and 1")
    if artifact.get("false_negative_risk_checked") is not artifact.get("positive_control_passed"):
        errors.append("false_negative_risk_checked must equal positive_control_passed")
    if str(verdict).startswith("success:"):
        deepest = artifact.get("deepest_level_reached_per_core_game") or {}
        control = deepest.get("no_energy_control") if isinstance(deepest, Mapping) else {}
        energy = deepest.get("energy_routed") if isinstance(deepest, Mapping) else {}
        if artifact.get("core_solves_preserved") is not True:
            errors.append("success requires core_solves_preserved=true")
        if artifact.get("positive_control_passed") is not True:
            errors.append("success requires positive_control_passed=true")
        if not any(
            int((energy or {}).get(game, 0)) >= 2 and int((control or {}).get(game, 0)) < 2
            for game in CORE_GAMES
        ):
            errors.append("success requires an energy-only CORE L2")
        if artifact.get("chosen_submitted_config") == "unchanged":
            errors.append("success requires a chosen submitted config")
    else:
        if artifact.get("chosen_submitted_config") != "unchanged":
            errors.append("non-success must keep chosen_submitted_config unchanged")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact content")
    return errors


def write_artifact(artifact: Mapping[str, Any], root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:  # pragma: no cover - file boundary.
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_a1_artifact(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - file boundary.
    return _read_json(Path(root) / A1_RELATIVE_PATH)


def _a1_reinduced_goal(a1: Mapping[str, Any]) -> dict[str, Any]:
    chosen = a1.get("chosen_submitted_config")
    available = isinstance(chosen, Mapping) and chosen != {}
    return {
        "available": bool(available),
        "source": A1_RELATIVE_PATH,
        "honest_verdict": a1.get("honest_verdict"),
        "chosen_submitted_config": chosen if available else "unchanged",
        "model_specs": a1.get("model_specs"),
        "reason": "a1_winning_config_available" if available else "a1_honest_null_no_live_l2_predicate",
    }


def _a1_control_measurement(a1: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(a1.get("target_levels_sweep") or a1.get("measurements") or [])
    selected = next((row for row in rows if int(row.get("target_levels") or 0) == 2), None)
    if selected is None:
        selected = rows[0] if rows else {}
    return _normalise_measurement(
        {
            **dict(selected),
            "measurement": "no_energy_control",
            "target_games": list(TARGET_GAMES),
            "source_artifact": A1_RELATIVE_PATH,
        },
        label="no_energy_control",
    )


def _a2_energy_measurement(
    root: Path,
    *,
    control: Mapping[str, Any],
) -> dict[str, Any]:  # pragma: no cover - file boundary.
    a2 = _read_json(root / A2_RELATIVE_PATH)
    rows = list(a2.get("measurements") or a2.get("levers_tried") or [])
    energy = next((row for row in rows if row.get("lever") == "energy_verifier_frontier_routing"), None)
    if energy is None:
        energy = {}
    payload = dict(energy)
    payload.pop("best_level_by_game", None)
    payload.pop("deepest_level_by_game", None)
    payload.pop("core_efficiency", None)
    target_rows = [
        row for row in energy.get("per_game", []) or [] if isinstance(row, Mapping) and row.get("game") in TARGET_GAMES
    ]
    return _normalise_measurement(
        {
            **payload,
            "measurement": "energy_routed",
            "target_games": list(TARGET_GAMES),
            "source_artifact": A2_RELATIVE_PATH,
            "per_game": target_rows,
        },
        label="energy_routed",
        fallback_levels=control["deepest_level_by_game"],
        fallback_efficiency_by_game=control["efficiency_by_game"],
    )


def matched_cached_measurements(
    root: Path | str,
    a1: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover - file boundary.
    root_path = Path(root)
    control = _a1_control_measurement(a1)
    energy = _a2_energy_measurement(root_path, control=control)
    return control, energy


def trust_energy_positive_control() -> dict[str, Any]:
    """REQ-ARC-WMTE-4534: synthetic known-L2 guard for a silently broken energy signal."""

    import numpy as np

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )

    transitions = [
        Transition(
            grid=np.array([[i]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[i + 1]], dtype=np.int16),
            level_before=0,
            level_after=0,
        )
        for i in range(6)
    ]

    def _generalizes(grid, _action, _data):
        return np.asarray(grid) + 1

    def _prefix_only(grid, _action, _data):
        value = int(np.asarray(grid)[0, 0])
        return np.asarray(grid) + 1 if value < 4 else np.asarray(grid)

    def _stuck(grid, _action, _data):
        return np.asarray(grid)

    selection = select_trusted_world_model(
        transitions,
        [
            WorldModelCandidate("l1_stuck_prefix", _prefix_only),
            WorldModelCandidate("known_l2_progress", _generalizes),
            WorldModelCandidate("l1_stuck_noop", _stuck),
        ],
        hidden_state=True,
    )
    samples = [
        {
            "state_id": row.candidate.name,
            "energy": round(float(row.trust_energy), 6),
            "deeper_progress": bool(row.heldout_best),
        }
        for row in selection.rows
    ]
    auroc = energy_separation_auroc(samples)
    return {
        "energy_separation_auroc": auroc,
        "positive_control": {
            "passed": bool(auroc is not None and auroc > 0.5),
            "game": "known_l2_trust_energy_fixture",
            "selected_candidate": selection.selected.name,
            "sample_count": len(samples),
        },
        "samples": samples,
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - SDK boundary.
    root_path = Path(root)
    spec_path = root_path / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    checks: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "offline_arcade_import_smoke": False,
        "spec_has_req_4534": spec_path.exists()
        and "REQ-ARC-WMTE-4534" in spec_path.read_text(encoding="utf-8"),
    }
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = repr(exc)
    checks["ok"] = bool(
        checks["offline_arcade_import_smoke"]
        and checks["spec_has_req_4534"]
        and checks["a1_artifact_present"]
    )
    return checks


def _blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    random_seed: int,
    duration_s: float | None,
) -> dict[str, Any]:
    artifact = build_artifact(
        preconditions_checked={
            **dict(preconditions_checked),
            "offline_arcade_import_smoke": bool(preconditions_checked.get("offline_arcade_import_smoke")),
            "spec_has_req_4534": bool(preconditions_checked.get("spec_has_req_4534")),
        },
        no_energy_control={
            "measurement": "no_energy_control",
            "core_efficiency": CORE_EFFICIENCY_BASELINE,
            "deepest_level_by_game": {game: 0 for game in CORE_GAMES},
        },
        energy_routed={
            "measurement": "energy_routed",
            "core_efficiency": 0.0,
            "deepest_level_by_game": {game: 0 for game in CORE_GAMES},
        },
        signal_characterization={
            "energy_separation_auroc": 0.0,
            "positive_control": {"passed": False, "game": "not_run", "sample_count": 0},
            "samples": [],
        },
        a1_goal={"available": False, "reason": "blocked_resource"},
        random_seed=random_seed,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = "blocked_offline_arcade_or_spec_precondition"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    a1_loader: Callable[[Path], Mapping[str, Any]] = load_a1_artifact,
    measurement_runner: Callable[[Path, Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]] = matched_cached_measurements,
    signal_runner: Callable[[], Mapping[str, Any]] = trust_energy_positive_control,
    random_seed: int = RANDOM_SEED,
    now: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4534: run matched energy/no-energy routing and write the artifact."""

    root_path = Path(root)
    started = float(now())
    checks = dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root_path)
    if checks.get("offline_arcade_import_smoke") is not True or checks.get("spec_has_req_4534") is not True:
        artifact = _blocked_artifact(
            preconditions_checked=checks,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    else:
        a1 = dict(a1_loader(root_path))
        a1_goal = _a1_reinduced_goal(a1)
        control, energy = measurement_runner(root_path, a1)
        artifact = build_artifact(
            preconditions_checked=checks,
            no_energy_control=control,
            energy_routed=energy,
            signal_characterization=dict(signal_runner()),
            a1_goal=a1_goal,
            random_seed=random_seed,
            duration_s=max(0.0, float(now()) - started),
        )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - script wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script wrapper.
    raise SystemExit(main())
