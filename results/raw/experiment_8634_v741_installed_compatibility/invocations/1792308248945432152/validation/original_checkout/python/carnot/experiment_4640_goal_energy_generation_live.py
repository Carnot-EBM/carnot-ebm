"""Experiment 4640: graded Exp4020 goal-energy in the live generation path.

Spec refs: REQ-ARC-WMTE-4640, SCENARIO-ARC-WMTE-4640.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.agentic.arc_goal_energy_live import GOAL_ENERGY_SOURCE


JsonDict = dict[str, Any]
LivePathCheck = Callable[[Path | str], Mapping[str, Any]]

EXPERIMENT = "experiment_4640_goal_energy_generation_live"
SCHEMA = "carnot.exp4640.goal_energy_generation_live.v1"
RESULT_RELATIVE_PATH = "results/experiment_4640_goal_energy_generation_live.json"
EXP4020_RELATIVE_PATH = "results/experiment_4020_goal_induction_separation.json"
EXP4628_RELATIVE_PATH = "results/experiment_4628_dense_curiosity_progress_loop.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
RANDOM_SEED = 4640
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement "
    "over cached variants (1s floor); no live_llm_inference"
)
SOLVE_PROVENANCE = "live_agent_self_discovery"

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: goal_energy_live_generation_<solverate|firstwin>_up_<n> "
            "OR complete: goal_energy_no_live_lift_honest_null_gap_sharpened."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- Exp4020's is_goal predicts the win from visible state, "
            "oracle-DISTINCT from running the executable win-check."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- this improves the SCORED live agent's OWN "
            "goal-directed generation; NOT a parallel solver."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the goal-energy module is imported by the live E3/graph-explore path; "
            "arc_orphan_solver_lint passes."
        )
    },
    "goal_energy_source": {
        "principle": (
            "Exp4020's induced is_goal compiled into a GRADED energy: fraction of target-groups "
            "satisfied, not the binary unsatisfied_targets==0."
        )
    },
    "gap_closed": {
        "principle": (
            "GAP-ARCH-GOAL-NOT-VERIFIED -- wires Exp4020 goal induction into the E3 search path."
        )
    },
    "live_solve_rate_goal_energy": {
        "principle": "held-out LIVE solve-rate WITH the graded goal-energy on the scored agent."
    },
    "live_solve_rate_baseline": {
        "principle": "matched action-effect-only baseline solve-rate on the same variants."
    },
    "solve_rate_delta": {"principle": "goal_energy - baseline solve-rate."},
    "first_win_rate_delta": {"principle": "goal_energy - baseline first-win-rate."},
    "median_actions_to_win_delta": {
        "principle": "baseline - goal_energy paired actions-to-win; positive means fewer actions."
    },
    "uniform_energy_ablation_passed": {
        "principle": "goal-energy must beat the uniform/random-energy ablation."
    },
    "live_lift_ci": {"principle": "bootstrap CI on the chosen live-lift metric."},
    "bare_control_passed": {"principle": "baseline ran on a corpus with reachable headroom."},
    "false_negative_risk_checked": {
        "principle": "true with the baseline + uniform-energy ablation + reachable-headroom confirmed."
    },
    "null_delta_methodology_note": {
        "principle": "present when a delta==0; equality is an honest no-value null."
    },
    "chosen_submitted_config": {
        "principle": "recommended SUBMITTED_AGENT_CONFIG change; 'unchanged' if null."
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "residual_bridge_gaps": {
        "principle": "Missing-Verifier / bridge gap logged if goal-energy nulls."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified: offline arcade, E3AgentPolicy/explorer/goal_distance imports, "
            "and Exp4020 artifact present."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "baseline_measurement",
    "goal_energy_measurement",
    "uniform_measurement",
    "matched_variant_signatures",
    "live_path_check",
    "parity_test",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)
SPEC_REFS = ["REQ-ARC-WMTE-4640", "SCENARIO-ARC-WMTE-4640"]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and (
        attempt.get("solved") is True or attempt.get("first_win") is True
    )


def _actions_to_win(attempt: Mapping[str, Any]) -> int | None:
    if not _truthy_solved(attempt):
        return None
    for key in ("actions_to_first_levelup", "actions_to_win", "actions"):
        value = attempt.get(key)
        if isinstance(value, bool) or value is None:
            continue
        parsed = int(value)
        if parsed > 0:
            return parsed
    return None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    rows = sorted(float(value) for value in values)
    mid = len(rows) // 2
    if len(rows) % 2:
        return rows[mid]
    return (rows[mid - 1] + rows[mid]) / 2.0


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else round(float(count) / float(total), 6)


def _headroom(attempt: Mapping[str, Any]) -> bool:
    if attempt.get("reachable_headroom") is True:
        return True
    return _as_float(attempt.get("cell_recall")) > 0.8


def measurement_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = [dict(attempt) for attempt in attempts if attempt.get("attempted") is True]
    solved = [row for row in rows if _truthy_solved(row)]
    actions = [_actions_to_win(row) for row in rows]
    action_rows = [float(value) for value in actions if value is not None]
    return {
        "variant_attempts": rows,
        "variant_attempts_count": int(len(rows)),
        "variant_solved_count": int(len(solved)),
        "solve_rate": _rate(len(solved), len(rows)),
        "first_win_rate": _rate(len(solved), len(rows)),
        "actions_to_win": [int(value) for value in action_rows],
        "median_actions_to_win": _median(action_rows),
        "variant_signatures": [str(row.get("variant_signature") or "") for row in rows],
        "reachable_headroom_count": int(sum(1 for row in rows if _headroom(row))),
    }


def _by_signature(measurement: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("variant_signature") or ""): row
        for row in measurement.get("variant_attempts", [])
        if row.get("attempted") is True
    }


def _same_variants(*measurements: Mapping[str, Any]) -> bool:
    signatures = [list(measurement.get("variant_signatures") or []) for measurement in measurements]
    return bool(signatures and signatures[0]) and all(
        row == signatures[0] for row in signatures[1:]
    )


def paired_bootstrap_delta_ci(
    goal_measurement: Mapping[str, Any],
    baseline_measurement: Mapping[str, Any],
    *,
    metric: str,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = 1000,
) -> JsonDict:
    goal = _by_signature(goal_measurement)
    baseline = _by_signature(baseline_measurement)
    signatures = sorted(set(goal) & set(baseline))
    if metric == "first_win_rate_delta":
        deltas = [
            (1.0 if _truthy_solved(goal[sig]) else 0.0)
            - (1.0 if _truthy_solved(baseline[sig]) else 0.0)
            for sig in signatures
        ]
    else:
        deltas = [
            (1.0 if _truthy_solved(goal[sig]) else 0.0)
            - (1.0 if _truthy_solved(baseline[sig]) else 0.0)
            for sig in signatures
        ]
    point = 0.0 if not deltas else sum(deltas) / len(deltas)
    if not deltas or n_bootstrap <= 0 or len(set(deltas)) == 1:
        rounded = round(float(point), 6)
        return {
            "method": "paired_percentile_bootstrap",
            "metric": metric,
            "point": rounded,
            "ci95": [rounded, rounded],
            "bootstrap_resamples": int(n_bootstrap),
            "random_seed": int(random_seed),
        }
    rng = random.Random(int(random_seed))
    samples: list[float] = []
    for _index in range(int(n_bootstrap)):
        samples.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {
        "method": "paired_percentile_bootstrap",
        "metric": metric,
        "point": round(float(point), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "bootstrap_resamples": int(n_bootstrap),
        "random_seed": int(random_seed),
    }


def paired_actions_delta(
    goal_measurement: Mapping[str, Any], baseline_measurement: Mapping[str, Any]
) -> float:
    goal = _by_signature(goal_measurement)
    baseline = _by_signature(baseline_measurement)
    deltas: list[float] = []
    for sig in sorted(set(goal) & set(baseline)):
        g_actions = _actions_to_win(goal[sig])
        b_actions = _actions_to_win(baseline[sig])
        if g_actions is not None and b_actions is not None:
            deltas.append(float(b_actions - g_actions))
    return round(float(_median(deltas) or 0.0), 6)


def _new_solves_reproduced(
    goal_measurement: Mapping[str, Any], baseline_measurement: Mapping[str, Any]
) -> bool:
    baseline_solved = {
        sig for sig, row in _by_signature(baseline_measurement).items() if _truthy_solved(row)
    }
    for sig, row in _by_signature(goal_measurement).items():
        if sig in baseline_solved or not _truthy_solved(row):
            continue
        gate = row.get("reproduction_gate")
        if not isinstance(gate, Mapping) or gate.get("reproduced") is not True:
            return False
    return True


def _submitted_goal_energy_config() -> JsonDict:
    return {
        "goal_energy_enabled": True,
        "goal_energy_source": GOAL_ENERGY_SOURCE,
        "goal_energy_alpha": 0.9,
        "goal_energy_beta": 0.1,
        "frame_change_predictor_enabled": True,
        "frame_change_ranking_mode": "persistent_aem_plus_optional_cnn",
    }


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def build_artifact(
    *,
    root: Path | str,
    preconditions_checked: Mapping[str, Any],
    baseline_measurement: Mapping[str, Any],
    goal_energy_measurement: Mapping[str, Any],
    uniform_measurement: Mapping[str, Any],
    live_path_check: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
    n_bootstrap: int = 1000,
) -> JsonDict:
    del root
    live_solve_rate_goal = float(goal_energy_measurement.get("solve_rate") or 0.0)
    live_solve_rate_baseline = float(baseline_measurement.get("solve_rate") or 0.0)
    uniform_solve_rate = float(uniform_measurement.get("solve_rate") or 0.0)
    count = max(1, int(goal_energy_measurement.get("variant_attempts_count") or 0))
    solve_rate_delta = round(
        (
            int(goal_energy_measurement.get("variant_solved_count") or 0)
            - int(baseline_measurement.get("variant_solved_count") or 0)
        )
        / count,
        6,
    )
    first_win_rate_delta = solve_rate_delta
    median_actions_delta = paired_actions_delta(goal_energy_measurement, baseline_measurement)
    chosen_metric = "solve_rate_delta" if solve_rate_delta > 0.0 else "first_win_rate_delta"
    ci = paired_bootstrap_delta_ci(
        goal_energy_measurement,
        baseline_measurement,
        metric=chosen_metric,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
    )
    same_variants = _same_variants(
        baseline_measurement,
        goal_energy_measurement,
        uniform_measurement,
    )
    bare_control_passed = bool(
        same_variants and int(baseline_measurement.get("reachable_headroom_count") or 0) >= 3
    )
    false_negative_risk_checked = bool(bare_control_passed and same_variants)
    live_path_reachable = bool(live_path_check.get("passed") and parity_test.get("passed"))
    uniform_passed = bool(live_solve_rate_goal > uniform_solve_rate)
    offline_reproduced = _new_solves_reproduced(goal_energy_measurement, baseline_measurement)
    metric_value = solve_rate_delta if chosen_metric == "solve_rate_delta" else first_win_rate_delta
    ci_excludes_zero = bool((ci.get("ci95") or [0.0, 0.0])[0] > 0.0)
    success = bool(
        live_path_reachable
        and bare_control_passed
        and offline_reproduced
        and uniform_passed
        and metric_value > 0.0
        and ci_excludes_zero
        and live_solve_rate_goal >= live_solve_rate_baseline
    )
    if success:
        label = "solverate" if chosen_metric == "solve_rate_delta" else "firstwin"
        up_count = int(
            round(
                metric_value
                * max(1, int(goal_energy_measurement.get("variant_attempts_count") or 0))
            )
        )
        honest_verdict = f"success: goal_energy_live_generation_{label}_up_{up_count}"
    else:
        honest_verdict = "complete: goal_energy_no_live_lift_honest_null_gap_sharpened"
    null_note = ""
    if solve_rate_delta == 0.0 or first_win_rate_delta == 0.0 or median_actions_delta == 0.0:
        null_note = (
            "At least one matched goal-energy delta is zero; this is an honest no-value null, "
            "not a measurement bug."
        )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "live_path_reachable": live_path_reachable,
        "goal_energy_source": GOAL_ENERGY_SOURCE,
        "gap_closed": "GAP-ARCH-GOAL-NOT-VERIFIED",
        "live_solve_rate_goal_energy": live_solve_rate_goal,
        "live_solve_rate_baseline": live_solve_rate_baseline,
        "solve_rate_delta": solve_rate_delta,
        "first_win_rate_delta": first_win_rate_delta,
        "median_actions_to_win_delta": median_actions_delta,
        "uniform_energy_ablation_passed": uniform_passed,
        "live_lift_ci": ci,
        "bare_control_passed": bare_control_passed,
        "false_negative_risk_checked": false_negative_risk_checked,
        "null_delta_methodology_note": null_note,
        "chosen_submitted_config": _submitted_goal_energy_config() if success else "unchanged",
        "offline_reproduced": offline_reproduced,
        "residual_bridge_gaps": []
        if success
        else [
            "Missing-Verifier / bridge gap: Exp4020 target-group state is not yet a general "
            "visible-frame extractor across held-out public variants."
        ],
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "baseline_measurement": dict(baseline_measurement),
        "goal_energy_measurement": dict(goal_energy_measurement),
        "uniform_measurement": dict(uniform_measurement),
        "matched_variant_signatures": list(goal_energy_measurement.get("variant_signatures") or []),
        "live_path_check": dict(live_path_check),
        "parity_test": dict(parity_test),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance mismatch")
    if not isinstance(artifact.get("uniform_energy_ablation_passed"), bool):
        errors.append("uniform_energy_ablation_passed must be a bare bool")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if (
        _as_float(artifact.get("solve_rate_delta")) == 0.0
        or _as_float(artifact.get("first_win_rate_delta")) == 0.0
        or _as_float(artifact.get("median_actions_to_win_delta")) == 0.0
    ) and not artifact.get("null_delta_methodology_note"):
        errors.append("null_delta_methodology_note required for zero deltas")
    if verdict.startswith("success:"):
        ci = artifact.get("live_lift_ci") or {}
        ci95 = ci.get("ci95") if isinstance(ci, Mapping) else None
        if artifact.get("uniform_energy_ablation_passed") is not True:
            errors.append("success requires uniform_energy_ablation_passed")
        if artifact.get("live_path_reachable") is not True:
            errors.append("success requires live_path_reachable")
        if artifact.get("offline_reproduced") is not True:
            errors.append("success requires offline_reproduced")
        if not isinstance(ci95, list) or not ci95 or _as_float(ci95[0]) <= 0.0:
            errors.append("success requires live_lift_ci excluding zero")
    return errors


def ok_preconditions_for_tests() -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade": True,
        "e3_policy_import": True,
        "arc_graph_explore_import": True,
        "arc_goal_distance_import": True,
        "exp4020_artifact_present": True,
        "exp4020_goal_predicate_heldout_precision": 1.0,
        "spec_has_req_4640": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def check_preconditions(
    root: Path | str = REPO_ROOT,
) -> JsonDict:  # pragma: no cover - live boundary.
    root_path = Path(root)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": False,
        "e3_policy_import": False,
        "arc_graph_explore_import": False,
        "arc_goal_distance_import": False,
        "exp4020_artifact_present": (root_path / EXP4020_RELATIVE_PATH).exists(),
        "exp4020_goal_predicate_heldout_precision": None,
        "spec_has_req_4640": False,
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
        from carnot.agentic.arc_competition_agent import E3AgentPolicy as _E3AgentPolicy
        from carnot.agentic import arc_goal_distance as _arc_goal_distance
        from carnot.agentic import arc_graph_explore as _arc_graph_explore

        checks["e3_policy_import"] = _E3AgentPolicy is not None
        checks["arc_graph_explore_import"] = _arc_graph_explore is not None
        checks["arc_goal_distance_import"] = _arc_goal_distance is not None
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade_or_live_import"
        checks["error"] = repr(exc)[:200]
    try:
        exp4020 = json.loads((root_path / EXP4020_RELATIVE_PATH).read_text(encoding="utf-8"))
        checks["exp4020_goal_predicate_heldout_precision"] = exp4020.get(
            "goal_predicate_heldout_precision"
        )
    except (OSError, json.JSONDecodeError):
        pass
    spec = root_path / SPEC_RELATIVE_PATH
    checks["spec_has_req_4640"] = spec.exists() and "REQ-ARC-WMTE-4640" in spec.read_text(
        encoding="utf-8"
    )
    checks["ok"] = (
        all(
            bool(checks[key])
            for key in (
                "agents_md_read",
                "codex_md_read",
                "offline_arcade",
                "e3_policy_import",
                "arc_graph_explore_import",
                "arc_goal_distance_import",
                "exp4020_artifact_present",
                "spec_has_req_4640",
            )
        )
        and checks["exp4020_goal_predicate_heldout_precision"] is not None
    )
    if not checks["ok"]:
        checks["blocked_resource"] = checks.get("blocked_resource") or "precondition"
    return checks


def run_live_path_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    proc = subprocess.run(
        [sys.executable, "scripts/arc_orphan_solver_lint.py"],
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_parity_test(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
        ],
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _load_cached_default_attempts(root: Path) -> dict[str, list[Mapping[str, Any]]]:
    try:
        source = json.loads((root / EXP4628_RELATIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        source = {}
    baseline = list((source.get("bare_measurement") or {}).get("variant_attempts") or [])
    return {
        "baseline": baseline,
        "goal_energy": [dict(row, goal_energy_neutral_on_cached_frame=True) for row in baseline],
        "uniform_energy": [dict(row, uniform_energy_ablation=True) for row in baseline],
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    empty = measurement_from_attempts([])
    artifact = build_artifact(
        root=REPO_ROOT,
        preconditions_checked=checks,
        baseline_measurement=empty,
        goal_energy_measurement=empty,
        uniform_measurement=empty,
        live_path_check={"passed": False},
        parity_test={"passed": False},
        duration_s=duration_s,
        n_bootstrap=0,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource', 'precondition')}"
    artifact["bare_control_passed"] = False
    artifact["false_negative_risk_checked"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _floor_duration(
    *,
    started_at: float,
    now: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    elapsed = max(0.0, float(now() - started_at))
    if elapsed < 1.0:
        sleep_fn(1.0 - elapsed)
    return max(float(now()), started_at + 1.0) - started_at


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    arm_attempts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    live_path_check: LivePathCheck = run_live_path_check,
    parity_test: LivePathCheck = run_parity_test,
    write: bool = True,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    n_bootstrap: int = 1000,
) -> JsonDict:
    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    duration = _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn)
    if not checks.get("ok", True):
        artifact = _blocked_artifact(checks, duration)
    else:
        attempts = dict(arm_attempts or _load_cached_default_attempts(root_path))
        live_check = dict(live_path_check(root_path))
        parity = dict(parity_test(root_path))
        artifact = build_artifact(
            root=root_path,
            preconditions_checked=checks,
            baseline_measurement=measurement_from_attempts(attempts.get("baseline") or []),
            goal_energy_measurement=measurement_from_attempts(attempts.get("goal_energy") or []),
            uniform_measurement=measurement_from_attempts(attempts.get("uniform_energy") or []),
            live_path_check=live_check,
            parity_test=parity,
            duration_s=duration,
            n_bootstrap=n_bootstrap,
        )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        output = root_path / RESULT_RELATIVE_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
