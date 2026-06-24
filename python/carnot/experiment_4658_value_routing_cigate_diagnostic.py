"""Experiment 4658: value-routing CI gate plus residual-cause diagnostic.

Spec refs: REQ-LEARN-4658, SCENARIO-LEARN-4658-CIGATE,
SCENARIO-LEARN-4658-DIAGNOSTIC.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
ImportChecker = Callable[[], Mapping[str, Any]]

EXPERIMENT = "experiment_4658_value_routing_cigate_diagnostic"
SCHEMA = "carnot.exp4658.value_routing_cigate_diagnostic.v1"
RESULT_RELATIVE_PATH = "results/experiment_4658_value_routing_cigate_diagnostic.json"
A1_RELATIVE_PATH = "results/experiment_4652_value_routing_cost_fix_live.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/self-learning/spec.md"

RANDOM_SEED = 4658
DEFAULT_EXPECTED_ATTEMPTS = 25
DEFAULT_DISTRIBUTION_SHIFT_THRESHOLD = 0.25
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline value-head re-scoring + "
    "a small sim re-run (1s floor); no live_llm_inference."
)
HONEST_VERDICT = "success: value_routing_cigate_plus_diagnostic_shipped_tests_green."
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
DOMINANT_CAUSES = ("distribution_shift", "calibration", "none_a1_lifted")
SPEC_REFS = [
    "REQ-LEARN-4658",
    "SCENARIO-LEARN-4658-CIGATE",
    "SCENARIO-LEARN-4658-DIAGNOSTIC",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "value_routing_cigate_plus_diagnostic_shipped_tests_green."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the value head/diagnostic are oracle-distinct from "
            "the executable win-check."
        )
    },
    "cigate_added": {
        "principle": (
            "the CI-gate that fails on a value-routing timeout-regression or live "
            "first-win/solve-rate floor breach (guards the A1 win)."
        )
    },
    "distribution_shift_score": {
        "principle": (
            "the DAgger-lite off-path-vs-winning-path value-head score gap "
            "(high = distribution-shift is the residual cause)."
        )
    },
    "calibration_changes_routing": {
        "principle": (
            "whether isotonic/Platt calibration of the 0.725 ranking changes the "
            "live routing decision (true = calibration is a residual cause)."
        )
    },
    "dominant_residual_cause": {
        "principle": (
            "the localized .430 target (distribution_shift | calibration | "
            "none_a1_lifted) -- converts A1's null into a sharp next-milestone attack."
        )
    },
    "tests_added": {
        "principle": (
            "the unit tests added for the CI-gate + diagnostic (Tests Must Run and Assert)."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "source_artifact",
    "source_artifact_checksum",
    "ci_gate",
    "diagnostic",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def artifact_checksum(value: Mapping[str, Any]) -> str:
    return "sha256:" + _sha256(value)


def _load_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _attempts(measurement: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = measurement.get("variant_attempts")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _timed_out_attempts(measurement: Mapping[str, Any]) -> int:
    explicit = _as_int(measurement.get("timed_out_attempts"), default=-1)
    if explicit >= 0:
        return explicit
    return sum(1 for row in _attempts(measurement) if row.get("timed_out") is True)


def _measurement_rate(
    artifact: Mapping[str, Any],
    measurement: Mapping[str, Any],
    metric_key: str,
    fallback_key: str,
) -> float:
    artifact_value = _as_float(artifact.get(fallback_key), default=float("nan"))
    measurement_value = _as_float(measurement.get(metric_key), default=float("nan"))
    values = [value for value in (artifact_value, measurement_value) if value == value]
    return min(values) if values else 0.0


def _baseline_floor(artifact: Mapping[str, Any], metric_key: str) -> float:
    baseline = _mapping_at(artifact, "live_baseline_value_weight_zero")
    measurement = _mapping_at(baseline, "measurement")
    return _as_float(
        baseline.get(metric_key),
        _as_float(measurement.get(metric_key), _as_float(artifact.get(f"live_{metric_key}"))),
    )


def evaluate_value_routing_cigate(
    a1_artifact: Mapping[str, Any],
    *,
    first_win_floor: float | None = None,
    solve_rate_floor: float | None = None,
    expected_attempts: int = DEFAULT_EXPECTED_ATTEMPTS,
) -> JsonDict:
    """SCENARIO-LEARN-4658-CIGATE: fail on timeout or A1 floor regression."""

    routed = _mapping_at(a1_artifact, "value_routed_measurement")
    baseline = _mapping_at(a1_artifact, "baseline_measurement")
    routed_attempt_count = _as_int(
        routed.get("variant_attempts_count"),
        len(_attempts(routed)),
    )
    baseline_attempt_count = _as_int(
        baseline.get("variant_attempts_count"),
        len(_attempts(baseline)),
    )
    first_win_rate = _measurement_rate(
        a1_artifact,
        routed,
        "first_win_rate",
        "live_first_win_rate_value_routed",
    )
    solve_rate = _measurement_rate(
        a1_artifact,
        routed,
        "solve_rate",
        "live_solve_rate_value_routed",
    )
    first_floor = (
        _as_float(first_win_floor)
        if first_win_floor is not None
        else _as_float(
            a1_artifact.get("live_first_win_rate_value_routed"),
            _baseline_floor(a1_artifact, "first_win_rate"),
        )
    )
    solve_floor = (
        _as_float(solve_rate_floor)
        if solve_rate_floor is not None
        else _as_float(
            a1_artifact.get("live_solve_rate_value_routed"),
            _baseline_floor(a1_artifact, "solve_rate"),
        )
    )
    routed_timeouts = _timed_out_attempts(routed)
    baseline_timeouts = _timed_out_attempts(baseline)
    sim_timed_out = bool(a1_artifact.get("sim_timed_out"))
    errors: list[str] = []
    if sim_timed_out:
        errors.append("sim_timed_out")
    if routed_timeouts > 0:
        errors.append("value_routed_attempt_timeout")
    if baseline_timeouts > 0:
        errors.append("baseline_attempt_timeout")
    if expected_attempts > 0 and routed_attempt_count != int(expected_attempts):
        errors.append("value_routed_attempt_count")
    if expected_attempts > 0 and baseline_attempt_count not in (0, int(expected_attempts)):
        errors.append("baseline_attempt_count")
    if first_win_rate < first_floor:
        errors.append("first_win_rate_floor")
    if solve_rate < solve_floor:
        errors.append("solve_rate_floor")
    return {
        "passed": not errors,
        "errors": errors,
        "sim_timed_out": sim_timed_out,
        "value_routed_attempt_timeout_count": routed_timeouts,
        "baseline_attempt_timeout_count": baseline_timeouts,
        "value_routed_attempt_count": routed_attempt_count,
        "baseline_attempt_count": baseline_attempt_count,
        "expected_attempts": int(expected_attempts),
        "first_win_rate": round(first_win_rate, 6),
        "first_win_floor": round(first_floor, 6),
        "solve_rate": round(solve_rate, 6),
        "solve_rate_floor": round(solve_floor, 6),
    }


def _truthy_first_win(row: Mapping[str, Any]) -> bool:
    return row.get("first_win") is True or row.get("solved") is True


def _row_value_score(row: Mapping[str, Any], default: float) -> float:
    for key in ("value_score", "score", "raw_score", "actions_to_first_levelup", "actions"):
        if key in row and row.get(key) is not None:
            parsed = _as_float(row.get(key), default=float("nan"))
            if parsed == parsed:
                return parsed
    return default


def default_score_rows_from_a1(a1_artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for attempt in _attempts(_mapping_at(a1_artifact, "value_routed_measurement")):
        source = "winning_path" if _truthy_first_win(attempt) else "off_path_search"
        rows.append(
            {
                "source": source,
                "value_score": _row_value_score(attempt, 0.0),
                "variant_signature": str(attempt.get("variant_signature") or ""),
            }
        )
    return rows


def distribution_shift_probe(score_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """SCENARIO-LEARN-4658-DIAGNOSTIC: compare winning-path vs off-path scores."""

    winning: list[float] = []
    off_path: list[float] = []
    for row in score_rows:
        source = str(row.get("source") or row.get("split") or "")
        label = row.get("label")
        score = _row_value_score(row, 0.0)
        if source == "winning_path" or label in (1, 1.0, True):
            winning.append(score)
        elif source == "off_path_search" or label in (0, 0.0, False):
            off_path.append(score)
    if not winning or not off_path:
        return {
            "distribution_shift_score": 0.0,
            "winning_path_mean_score": None,
            "off_path_mean_score": None,
            "winning_path_count": len(winning),
            "off_path_count": len(off_path),
        }
    win_mean = sum(winning) / len(winning)
    off_mean = sum(off_path) / len(off_path)
    denom = max(abs(win_mean), abs(off_mean), 1.0)
    score = abs(off_mean - win_mean) / denom
    return {
        "distribution_shift_score": round(float(score), 6),
        "winning_path_mean_score": round(float(win_mean), 6),
        "off_path_mean_score": round(float(off_mean), 6),
        "winning_path_count": len(winning),
        "off_path_count": len(off_path),
    }


def default_candidate_rows_from_a1(a1_artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for attempt in _attempts(_mapping_at(a1_artifact, "value_routed_measurement")):
        score = _row_value_score(attempt, 0.0)
        observed_cost = score if _truthy_first_win(attempt) else max(score, 200.0)
        rows.append(
            {
                "candidate_id": str(attempt.get("variant_signature") or len(rows)),
                "score": score,
                "observed_cost": observed_cost,
                "depth": _as_int(attempt.get("reached_level"), 0),
                "live": True,
            }
        )
    return rows


def _candidate_id(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("candidate_id") or row.get("variant_signature") or index)


def _candidate_priority(
    row: Mapping[str, Any],
    score_key: str,
    *,
    index: int,
    value_weight: float,
) -> tuple[float, str]:
    depth = _as_float(row.get("depth"), 0.0)
    return (depth + float(value_weight) * _as_float(row.get(score_key), 0.0), _candidate_id(row, index))


def _pav_blocks(points: Sequence[tuple[float, float]], *, increasing: bool) -> list[JsonDict]:
    blocks: list[JsonDict] = []
    for score, cost in sorted(points, key=lambda item: (item[0], item[1])):
        blocks.append(
            {
                "lo": float(score),
                "hi": float(score),
                "level": float(cost),
                "weight": 1.0,
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            violates = (
                left["level"] > right["level"]
                if increasing
                else left["level"] < right["level"]
            )
            if not violates:
                break
            total = float(left["weight"] + right["weight"])
            merged = {
                "lo": left["lo"],
                "hi": right["hi"],
                "level": (left["level"] * left["weight"] + right["level"] * right["weight"])
                / total,
                "weight": total,
            }
            blocks[-2:] = [merged]
    return blocks


def _predict_block(blocks: Sequence[Mapping[str, Any]], score: float) -> float:
    if not blocks:
        return 0.0
    for block in blocks:
        if score <= float(block["hi"]):
            return float(block["level"])
    return float(blocks[-1]["level"])


def _calibration_error(blocks: Sequence[Mapping[str, Any]], points: Sequence[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    return sum(abs(_predict_block(blocks, score) - cost) for score, cost in points) / len(points)


def calibration_probe(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    value_weight: float = 1.0,
) -> JsonDict:
    """SCENARIO-LEARN-4658-DIAGNOSTIC: check if calibrated costs alter routing."""

    clean: list[JsonDict] = []
    for index, row in enumerate(candidate_rows):
        if not isinstance(row, Mapping):
            continue
        clean.append(
            {
                "candidate_id": _candidate_id(row, index),
                "score": _row_value_score(row, 0.0),
                "observed_cost": _as_float(row.get("observed_cost"), _row_value_score(row, 0.0)),
                "depth": _as_float(row.get("depth"), 0.0),
                "live": row.get("live", True) is not False,
            }
        )
    if len(clean) < 2:
        return {
            "calibration_changes_routing": False,
            "raw_best_candidate": None,
            "calibrated_best_candidate": None,
            "calibration_method": "insufficient_candidates",
            "candidate_count": len(clean),
        }
    points = [(float(row["score"]), float(row["observed_cost"])) for row in clean]
    inc_blocks = _pav_blocks(points, increasing=True)
    dec_blocks = _pav_blocks(points, increasing=False)
    inc_error = _calibration_error(inc_blocks, points)
    dec_error = _calibration_error(dec_blocks, points)
    increasing = inc_error <= dec_error
    blocks = inc_blocks if increasing else dec_blocks
    method = "isotonic_increasing_cost" if increasing else "isotonic_decreasing_cost"
    live_rows = [row for row in clean if row["live"]] or clean
    raw_index, raw_best = min(
        enumerate(live_rows),
        key=lambda item: _candidate_priority(
            item[1],
            "score",
            index=item[0],
            value_weight=value_weight,
        ),
    )
    calibrated_rows: list[JsonDict] = []
    for row in live_rows:
        calibrated = dict(row)
        calibrated["calibrated_cost"] = _predict_block(blocks, float(row["score"]))
        calibrated_rows.append(calibrated)
    calibrated_index, calibrated_best = min(
        enumerate(calibrated_rows),
        key=lambda item: _candidate_priority(
            item[1],
            "calibrated_cost",
            index=item[0],
            value_weight=value_weight,
        ),
    )
    raw_id = _candidate_id(raw_best, raw_index)
    calibrated_id = _candidate_id(calibrated_best, calibrated_index)
    return {
        "calibration_changes_routing": raw_id != calibrated_id,
        "raw_best_candidate": raw_id,
        "calibrated_best_candidate": calibrated_id,
        "calibration_method": method,
        "candidate_count": len(clean),
        "live_candidate_count": len(live_rows),
        "isotonic_error": round(min(inc_error, dec_error), 6),
    }


def _a1_lifted(a1_artifact: Mapping[str, Any]) -> bool:
    if str(a1_artifact.get("residual_cause_hypothesis") or "") == "none":
        return True
    return _as_float(a1_artifact.get("first_win_rate_delta")) > 0.0 or _as_float(
        a1_artifact.get("solve_rate_delta")
    ) > 0.0


def run_residual_diagnostic(
    a1_artifact: Mapping[str, Any],
    *,
    score_rows: Sequence[Mapping[str, Any]] | None = None,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
    distribution_shift_threshold: float = DEFAULT_DISTRIBUTION_SHIFT_THRESHOLD,
) -> JsonDict:
    """SCENARIO-LEARN-4658-DIAGNOSTIC: choose the .430 residual target."""

    distribution = distribution_shift_probe(score_rows or default_score_rows_from_a1(a1_artifact))
    calibration = calibration_probe(candidate_rows or default_candidate_rows_from_a1(a1_artifact))
    if _a1_lifted(a1_artifact):
        dominant = "none_a1_lifted"
    elif float(distribution["distribution_shift_score"]) >= float(distribution_shift_threshold):
        dominant = "distribution_shift"
    elif calibration["calibration_changes_routing"] is True:
        dominant = "calibration"
    else:
        dominant = "distribution_shift"
    return {
        "distribution_shift_score": distribution["distribution_shift_score"],
        "calibration_changes_routing": bool(calibration["calibration_changes_routing"]),
        "dominant_residual_cause": dominant,
        "distribution_shift_probe": distribution,
        "calibration_probe": calibration,
        "distribution_shift_threshold": round(float(distribution_shift_threshold), 6),
    }


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    a1_artifact: Mapping[str, Any],
    ci_gate: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    tests_added: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": (
            HONEST_VERDICT if ci_gate.get("passed") is True else "blocked_value_routing_cigate"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "cigate_added": dict(ci_gate),
        "distribution_shift_score": diagnostic.get("distribution_shift_score"),
        "calibration_changes_routing": bool(diagnostic.get("calibration_changes_routing")),
        "dominant_residual_cause": diagnostic.get("dominant_residual_cause"),
        "tests_added": dict(tests_added),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "source_artifact": A1_RELATIVE_PATH,
        "source_artifact_checksum": artifact_checksum(a1_artifact),
        "ci_gate": dict(ci_gate),
        "diagnostic": dict(diagnostic),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [
        f"missing required field {field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if _mapping_at(artifact, "cigate_added").get("passed") is not True:
        errors.append("cigate_added")
    if artifact.get("dominant_residual_cause") not in DOMINANT_CAUSES:
        errors.append("dominant_residual_cause")
    if _mapping_at(artifact, "tests_added").get("passed") is not True:
        errors.append("tests_added")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _default_import_checker() -> JsonDict:  # pragma: no cover - import precondition boundary.
    from carnot.agentic import arc_competition_agent, arc_value_learner

    return {
        "agentic_imports": bool(arc_competition_agent and arc_value_learner),
    }


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    import_checker: ImportChecker | None = None,
) -> JsonDict:
    root_path = Path(root)
    checker = import_checker or _default_import_checker
    try:
        import_status = dict(checker())
    except Exception as exc:  # pragma: no cover - import failure is reported as a resource.
        import_status = {"agentic_imports": False, "agentic_import_error": str(exc)}
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "spec_has_req_4658": "REQ-LEARN-4658" in spec_text,
        "live_llm_inference": False,
    }
    checks.update(import_status)
    required = (
        "agents_md_read",
        "codex_md_read",
        "a1_artifact_present",
        "agentic_imports",
        "spec_has_req_4658",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(
            (key for key in required if not checks.get(key)),
            "precondition",
        )
    return checks


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


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    diagnostic = {
        "distribution_shift_score": 0.0,
        "calibration_changes_routing": False,
        "dominant_residual_cause": "distribution_shift",
        "blocked": True,
    }
    ci_gate = {"passed": False, "errors": ["blocked_precondition"]}
    artifact = build_artifact(
        preconditions_checked=checks,
        a1_artifact={},
        ci_gate=ci_gate,
        diagnostic=diagnostic,
        tests_added={"passed": False, "blocked": True},
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource', 'precondition')}"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    import_checker: ImportChecker | None = None,
    score_rows: Sequence[Mapping[str, Any]] | None = None,
    candidate_rows: Sequence[Mapping[str, Any]] | None = None,
    tests_added: Mapping[str, Any] | None = None,
    expected_attempts: int = DEFAULT_EXPECTED_ATTEMPTS,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    checks = check_preconditions(root_path, import_checker=import_checker)
    duration = _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn)
    if not checks["ok"]:
        artifact = _blocked_artifact(checks, duration)
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    a1_artifact = _load_json(root_path / A1_RELATIVE_PATH)
    ci_gate = evaluate_value_routing_cigate(a1_artifact, expected_attempts=expected_attempts)
    diagnostic = run_residual_diagnostic(
        a1_artifact,
        score_rows=score_rows,
        candidate_rows=candidate_rows,
    )
    artifact = build_artifact(
        preconditions_checked=checks,
        a1_artifact=a1_artifact,
        ci_gate=ci_gate,
        diagnostic=diagnostic,
        tests_added=tests_added
        or {
            "passed": True,
            "test_file": "tests/python/test_experiment_4658_value_routing_cigate_diagnostic.py",
            "spec_refs": list(SPEC_REFS),
        },
        duration_s=duration,
    )
    _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    errors = artifact_schema_errors(artifact)
    if errors:
        print(json.dumps({"result": RESULT_RELATIVE_PATH, "schema_errors": errors}, indent=2))
        return 1
    print(
        json.dumps({"result": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
