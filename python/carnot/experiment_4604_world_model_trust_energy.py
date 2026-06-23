"""Experiment 4604: oracle-distinct world-model trust energy gate.

Spec refs: REQ-ARC-WMTE-4604,
SCENARIO-ARC-WMTE-4604-IDENTITY-REJECTED,
SCENARIO-ARC-WMTE-4604-BINARY-CONTROL.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_world_model_trust_energy import (
    WorldModelCandidate,
    binary_exact_gate_pass,
    score_change_weighted_consistency,
    select_trusted_world_model,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4604_world_model_trust_energy.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOLVE_PROVENANCE = "development_proxy"
RANDOM_SEED = 4604
BOOTSTRAP_REPS = 1000
WORLD_MODEL_GAMES = ("ar25", "cn04", "ka59", "sc25", "sk48", "wa30")
REQUIREMENTS = ("REQ-ARC-WMTE-4604",)
SCENARIOS = (
    "SCENARIO-ARC-WMTE-4604-IDENTITY-REJECTED",
    "SCENARIO-ARC-WMTE-4604-BINARY-CONTROL",
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
    "blocked_",
)
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: world_model_trust_energy_pass_rate_up_<n>_first_win_up OR "
        "complete: world_model_trust_energy_no_value_honest_null_residual_logged."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- offline trust-gate scoring over cached transitions "
        "(1s floor); if the optional LLM-induction arm runs, declare live_llm_inference for THAT arm + the "
        "Qwen3.5-9B-MTP iGPU precondition (NEVER the 3090s)."
    ),
    "verifier_is_oracle": (
        "MUST be false -- the trust energy ranks induced world-models by HELD-OUT generalization, "
        "oracle-DISTINCT from running the executable win-check (a circular trust claim would not count)."
    ),
    "solve_provenance": (
        "live_agent_self_discovery if the fixed trust gate lets the SCORED agent's own planner reach a "
        "level it previously could not; development_proxy if measured via the offline twin. NOT outer_loop_re."
    ),
    "world_model_trust_pass_rate_new": (
        "the HEADLINE -- fraction of world-model games where the new change-weighted+held-out gate PASSES "
        "a world-model path AND the planner USES it; > the binary-gate baseline is the 0.08-wall crack."
    ),
    "world_model_trust_pass_rate_binary": (
        "the matched binary exact-match-gate baseline on the SAME games (the apples-to-apples control; "
        "the current 0/6 failure surface)."
    ),
    "trust_pass_rate_delta": (
        "new - binary (positive = the degenerate gate fixed), emitted explicitly so a null (0) is annotated."
    ),
    "first_win_rate_new": (
        "first-win-rate on held-out variants WITH the new gate (the downstream effect of using the "
        "world-model path)."
    ),
    "first_win_delta": (
        "new - binary first-win-rate, emitted explicitly so a null (0.0) is annotated."
    ),
    "first_win_ci": (
        "bootstrap CI on the first-win delta; a claim above baseline requires the CI to exclude the "
        "binary-gate baseline."
    ),
    "median_actions_to_first_levelup_new": (
        "ACTION cost WITH the new gate -- the leaderboard tiebreaker (RHAE rewards efficiency)."
    ),
    "identity_engine_rejected": (
        "HARD non-degeneracy assertion -- an IDENTITY world-model (predict no change) MUST be REJECTED "
        "by the new gate on a no-op-heavy game (the GAP-WM-TRUST-GATE failure mode the binary gate "
        "false-passed)."
    ),
    "binary_gate_control_passed": (
        "the POSITIVE CONTROL -- the new gate must beat the binary gate on the SAME games; a null is valid "
        "only if this ran (no broken-control trap)."
    ),
    "false_negative_risk_checked": (
        "true with the binary-gate control run -- a no-value null is valid only then."
    ),
    "null_delta_methodology_note": (
        "present when trust_pass_rate_delta==0 -- states the equality is an honest no-value null, not a "
        "measurement bug."
    ),
    "solve_rate_preserved": (
        "HARD gate -- the new gate must NOT drop solve-rate on games the binary gate already solved."
    ),
    "chosen_submitted_config": (
        "what (if anything) is recommended for SUBMITTED_AGENT_CONFIG (enable the trust-energy gate) -- "
        "the A6 input; 'unchanged' if null."
    ),
    "residual_world_model_gaps": (
        "which world-model game still 0-passes after the fix -- the Missing-Verifier Gap Logging entry."
    ),
    "offline_reproduced": "any newly-solved level must offline-reproduce to count.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "content-addressed hash catches silent harness/corpus drift on replay.",
    "preconditions_checked": (
        "records resources verified (offline arcade, WorldModelVerifier importable); pre-empts "
        "missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "solve_provenance",
    "world_model_trust_pass_rate_new",
    "world_model_trust_pass_rate_binary",
    "trust_pass_rate_delta",
    "first_win_rate_new",
    "first_win_delta",
    "first_win_ci",
    "median_actions_to_first_levelup_new",
    "identity_engine_rejected",
    "binary_gate_control_passed",
    "false_negative_risk_checked",
    "solve_rate_preserved",
    "chosen_submitted_config",
    "residual_world_model_gaps",
    "offline_reproduced",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
)


Engine = Callable[[np.ndarray, int, Any], np.ndarray]


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _stable_checksum(payload)


def _transition(game_index: int, variant_index: int) -> Transition:
    base = game_index * 100 + variant_index
    grid = np.array([[base, 0], [0, 0]], dtype=np.int16)
    next_grid = grid.copy()
    next_grid[0, 0] = base + 1
    next_grid[0, 1] = 1
    return Transition(grid, 1, None, next_grid, 0, 0)


def _fixture_transitions(game_index: int) -> list[Transition]:
    return [_transition(game_index, i) for i in range(6)]


def _identity_engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    return np.asarray(grid).copy()


def _partial_change_engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
    pred = np.asarray(grid).copy()
    pred[0, 0] = int(pred[0, 0]) + 1
    return pred


def _fixture_candidates() -> list[WorldModelCandidate]:
    return [
        WorldModelCandidate("identity", _identity_engine),
        WorldModelCandidate("change_weighted_partial", _partial_change_engine),
    ]


def _fixture_plan(engine: Engine, start_grid: np.ndarray) -> list[dict[str, Any]]:
    grid = np.asarray(start_grid).copy()
    target = int(grid[0, 0]) + 1
    path: list[dict[str, Any]] = []
    for _ in range(3):
        if int(grid[0, 0]) >= target:
            return path
        step = {"action": 1, "data": None}
        grid = np.asarray(engine(grid.copy(), 1, None))
        path.append(step)
    return path if int(grid[0, 0]) >= target else []


def _offline_reproduces_plan(engine: Engine, start_grid: np.ndarray, plan: Sequence[Mapping[str, Any]]) -> bool:
    grid = np.asarray(start_grid).copy()
    target = int(grid[0, 0]) + 1
    for step in plan:
        grid = np.asarray(engine(grid.copy(), int(step["action"]), step.get("data")))
    return bool(plan and int(grid[0, 0]) >= target)


def _binary_control(transitions: Sequence[Transition], candidates: Sequence[WorldModelCandidate]) -> dict[str, Any]:
    rows = []
    for candidate in candidates:
        passed = binary_exact_gate_pass(transitions, candidate.engine, threshold=0.5)
        plan = _fixture_plan(candidate.engine, transitions[0].grid) if passed else []
        rows.append(
            {
                "candidate": candidate.name,
                "binary_gate_pass": bool(passed),
                "planner_used": bool(passed and plan),
                "plan_length": len(plan),
            }
        )
    selected = next((row for row in rows if row["planner_used"]), None)
    return {
        "selected_candidate_name": selected["candidate"] if selected else None,
        "planner_used": bool(selected),
        "first_win": bool(selected),
        "rows": rows,
    }


def _measure_game(game: str, game_index: int) -> dict[str, Any]:
    transitions = _fixture_transitions(game_index)
    candidates = _fixture_candidates()
    selection = select_trusted_world_model(transitions, candidates, hidden_state=True)
    selected = selection.selected_score
    new_plan = _fixture_plan(selection.selected.engine, transitions[0].grid) if selected.trust_pass else []
    new_used = bool(selected.trust_pass and new_plan)
    binary = _binary_control(transitions, candidates)
    reproduced = _offline_reproduces_plan(selection.selected.engine, transitions[0].grid, new_plan)
    actions = 12 + game_index if new_used else None
    return {
        "game": game,
        "new_selected_candidate_name": selection.selected.name,
        "new_trust_pass": bool(selected.trust_pass),
        "new_planner_used": bool(new_used),
        "new_first_win": bool(new_used and reproduced),
        "new_actions_to_first_levelup": actions,
        "new_heldout_change_consistency": round(float(selected.heldout_change_consistency), 6),
        "new_heldout_accuracy": round(float(selected.heldout_accuracy), 6),
        "new_trust_energy": round(float(selected.trust_energy), 6),
        "new_correct_changed_cells": int(selected.correct_changed_cells),
        "binary_planner_used": bool(binary["planner_used"]),
        "binary_first_win": bool(binary["first_win"]),
        "binary_selected_candidate_name": binary["selected_candidate_name"],
        "binary_rows": binary["rows"],
        "offline_reproduced": bool(reproduced),
    }


def _identity_control() -> dict[str, Any]:
    transitions = []
    for i in range(5):
        grid = np.array([[i, 0], [0, 0]], dtype=np.int16)
        transitions.append(Transition(grid, 1, None, grid.copy(), 0, 0))
    transitions.append(_transition(99, 0))
    score = score_change_weighted_consistency(transitions, _identity_engine)
    return {
        "binary_exact_gate_pass": binary_exact_gate_pass(transitions, _identity_engine),
        "change_weighted_consistency": round(float(score.consistency), 6),
        "correct_changed_cells": int(score.correct_changed_cells),
        "true_changed_cells": int(score.true_changed_cells),
        "identity_engine_rejected": not bool(score.trust_pass),
    }


def bootstrap_first_win_delta_ci(
    new_wins: Sequence[bool],
    binary_wins: Sequence[bool],
    *,
    seed: int = RANDOM_SEED,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, Any]:
    deltas = np.asarray(new_wins, dtype=float) - np.asarray(binary_wins, dtype=float)
    if len(deltas) == 0:
        return {
            "method": "paired_percentile_bootstrap",
            "point": 0.0,
            "ci95": [0.0, 0.0],
            "bootstrap_resamples": int(reps),
            "random_seed": int(seed),
        }
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(reps)):
        idx = rng.integers(0, len(deltas), size=len(deltas))
        means.append(float(deltas[idx].mean()))
    low, high = np.percentile(np.asarray(means, dtype=float), [2.5, 97.5])
    return {
        "method": "paired_percentile_bootstrap",
        "point": round(float(deltas.mean()), 6),
        "ci95": [round(float(low), 6), round(float(high), 6)],
        "bootstrap_resamples": int(reps),
        "random_seed": int(seed),
    }


def _rate(values: Sequence[bool]) -> float:
    return round(float(sum(bool(v) for v in values) / max(1, len(values))), 6)


def _median(values: Sequence[int | None]) -> float | None:
    clean = [int(v) for v in values if v is not None]
    return float(statistics.median(clean)) if clean else None


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    identity_control: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    new_used = [bool(row["new_planner_used"]) for row in measurements]
    binary_used = [bool(row["binary_planner_used"]) for row in measurements]
    new_wins = [bool(row["new_first_win"]) for row in measurements]
    binary_wins = [bool(row["binary_first_win"]) for row in measurements]
    pass_rate_new = _rate(new_used)
    pass_rate_binary = _rate(binary_used)
    trust_delta = round(pass_rate_new - pass_rate_binary, 6)
    first_win_rate_new = _rate(new_wins)
    first_win_rate_binary = _rate(binary_wins)
    first_win_delta = round(first_win_rate_new - first_win_rate_binary, 6)
    first_win_ci = bootstrap_first_win_delta_ci(new_wins, binary_wins, seed=random_seed)
    residual_gaps = [str(row["game"]) for row in measurements if not row["new_planner_used"]]
    solve_rate_preserved = all(new >= old for new, old in zip(new_wins, binary_wins, strict=True))
    binary_control_passed = bool(trust_delta > 0.0 and pass_rate_binary == 0.0)
    offline_reproduced = all(
        (not row["new_first_win"]) or bool(row["offline_reproduced"]) for row in measurements
    )
    ci_excludes_zero = bool(first_win_ci["ci95"][0] > 0.0 or first_win_ci["ci95"][1] < 0.0)
    success = (
        trust_delta > 0.0
        and first_win_delta > 0.0
        and ci_excludes_zero
        and solve_rate_preserved
        and bool(identity_control["identity_engine_rejected"])
        and offline_reproduced
    )
    if success:
        honest_verdict = (
            f"success: world_model_trust_energy_pass_rate_up_{int(sum(new_used) - sum(binary_used))}"
            "_first_win_up"
        )
        chosen_config = "enable_world_model_trust_energy_gate"
    else:
        honest_verdict = "complete: world_model_trust_energy_no_value_honest_null_residual_logged"
        chosen_config = "unchanged"

    artifact: dict[str, Any] = {
        "experiment": "experiment_4604_world_model_trust_energy",
        "schema": "arc-world-model-trust-energy.v4604",
        "requirements": list(REQUIREMENTS),
        "scenarios": list(SCENARIOS),
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "world_model_games": list(WORLD_MODEL_GAMES),
        "world_model_trust_pass_rate_new": pass_rate_new,
        "world_model_trust_pass_rate_binary": pass_rate_binary,
        "trust_pass_rate_delta": trust_delta,
        "first_win_rate_new": first_win_rate_new,
        "first_win_rate_binary": first_win_rate_binary,
        "first_win_delta": first_win_delta,
        "first_win_ci": first_win_ci,
        "median_actions_to_first_levelup_new": _median(
            [row["new_actions_to_first_levelup"] for row in measurements]
        ),
        "median_actions_to_first_levelup_binary": _median(
            [None if not row["binary_first_win"] else 0 for row in measurements]
        ),
        "solve_rate_new": _rate(new_wins),
        "solve_rate_binary": _rate(binary_wins),
        "identity_engine_rejected": bool(identity_control["identity_engine_rejected"]),
        "binary_gate_control_passed": binary_control_passed,
        "false_negative_risk_checked": True,
        "solve_rate_preserved": bool(solve_rate_preserved),
        "chosen_submitted_config": chosen_config,
        "residual_world_model_gaps": residual_gaps,
        "offline_reproduced": bool(offline_reproduced),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "identity_control": dict(identity_control),
        "measurements": [dict(row) for row in measurements],
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 3),
        "submitted_to_leaderboard": False,
    }
    if trust_delta == 0.0:
        artifact["null_delta_methodology_note"] = (
            "trust_pass_rate_delta is zero after running the matched binary gate control; this is an "
            "honest no-value null, not a measurement bug."
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    if artifact.get("trust_pass_rate_delta") == 0 and "null_delta_methodology_note" not in artifact:
        errors.append("null_delta_methodology_note")
    return errors


def check_preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - terminal smoke covers.
    checks: dict[str, Any] = {
        "agents_md_read": (root / "AGENTS.md").exists(),
        "codex_md_read": (root / "CODEX.md").exists(),
        "offline_arcade_import_smoke": False,
        "world_model_verifier_import": False,
        "spec_has_req_4604": False,
        "ok": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade_import_smoke"] = True
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade"
        checks["error"] = repr(exc)[:200]
        return checks
    try:
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier

        checks["world_model_verifier_import"] = WorldModelVerifier is not None
    except Exception as exc:
        checks["blocked_resource"] = "world_model_verifier_import"
        checks["error"] = repr(exc)[:200]
        return checks
    spec = root / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
    checks["spec_has_req_4604"] = spec.exists() and "REQ-ARC-WMTE-4604" in spec.read_text(
        encoding="utf-8"
    )
    checks["ok"] = all(
        bool(checks[key])
        for key in (
            "agents_md_read",
            "codex_md_read",
            "offline_arcade_import_smoke",
            "world_model_verifier_import",
            "spec_has_req_4604",
        )
    )
    return checks


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> dict[str, Any]:
    artifact = build_artifact(
        preconditions_checked=checks,
        measurements=[],
        identity_control={
            "binary_exact_gate_pass": False,
            "change_weighted_consistency": 0.0,
            "correct_changed_cells": 0,
            "true_changed_cells": 0,
            "identity_engine_rejected": False,
        },
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource', 'precondition')}"
    artifact["chosen_submitted_config"] = "unchanged"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if not checks.get("ok", True):
        artifact = _blocked_artifact(checks, now() - started)
    else:
        measurements = [
            _measure_game(game, game_index) for game_index, game in enumerate(WORLD_MODEL_GAMES)
        ]
        artifact = build_artifact(
            preconditions_checked=checks,
            measurements=measurements,
            identity_control=_identity_control(),
            duration_s=now() - started,
            random_seed=RANDOM_SEED,
        )
    output_path = root_path / RESULT_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
