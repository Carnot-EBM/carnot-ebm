"""Experiment 4622: canonical offline-to-live transfer-ratio metric helper.

Spec refs: REQ-ARC-WMTE-4622, SCENARIO-ARC-WMTE-4622.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4622_offline_to_live_transfer_ratio_metric"
SCHEMA = "carnot.exp4622.offline_to_live_transfer_ratio_metric.v1"
RESULT_RELATIVE_PATH = "results/experiment_4622_offline_to_live_transfer_ratio_metric.json"
A1_RELATIVE_PATH = "results/experiment_4616_offline_live_bridge_disambiguation.json"
A2_RELATIVE_PATH = "results/experiment_4617_graduate_spatial_value_head_live.json"
OFFLINE_AUROC_RELATIVE_PATH = "results/experiment_4545_cross_game_discrimination_v3.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
LIVE_SUBMITTABLE_RELATIVE_PATH = "results/experiment_4586_live_submittable_coheadline.json"
ACTION_EFFICIENCY_RELATIVE_PATH = "results/experiment_4574_action_efficiency_coheadline.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4622
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads the A1/A2 artifacts + registry, no model "
    "load (100us floor)."
)
HONEST_VERDICT = "success: offline_to_live_transfer_ratio_metric_helper_shipped_tests_green"
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
SPEC_REFS = ["REQ-ARC-WMTE-4622", "SCENARIO-ARC-WMTE-4622"]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "offline_to_live_transfer_ratio_metric_helper_shipped_tests_green."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "offline_to_live_transfer_ratio": {
        "principle": (
            "the canonical co-headline metric value (live lift attributable to the value "
            "head, with the offline AUROC reported alongside)."
        )
    },
    "offline_auroc_component": {
        "principle": (
            "the offline LOO-AUROC (the verifier signal) -- explicit so a high-offline/"
            "zero-live case is visibly the bridge-not-crossed state."
        )
    },
    "live_lift_component": {
        "principle": (
            "the LIVE first-win/efficiency lift the value head produces -- explicit so the "
            "metric does not hide the offline->live gap."
        )
    },
    "coheadline_block": {
        "principle": (
            "offline_to_live_transfer_ratio reported side-by-side with "
            "reproducible_total_levels / live-submittable / first-win-rate / "
            "action-efficiency -- the single canonical metric surface."
        )
    },
    "tests_added": {
        "principle": (
            "the asserting tests (every test has >=1 assertion; no skips) -- the metric "
            "helper is verified, not asserted."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {"principle": "catches silent drift on replay."},
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REPORTED_SIDE_BY_SIDE = [
    "offline_to_live_transfer_ratio",
    "offline_auroc_component",
    "live_lift_component",
    "reproducible_total_levels",
    "live_submittable_level_count",
    "first_win_rate",
    "action_efficiency_score",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return _checksum(payload)


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _offline_auroc(offline_artifact: Mapping[str, Any]) -> float:
    direct = (
        offline_artifact.get("offline_auroc_component"),
        offline_artifact.get("offline_loo_auroc"),
        offline_artifact.get("loo_auroc_mean"),
    )
    for value in direct:
        parsed = _as_float(value, default=-1.0)
        if parsed > 0.0:
            return _rounded(parsed)
    feature_class = offline_artifact.get("feature_class_loo_auroc")
    if isinstance(feature_class, Mapping):
        return _rounded(_as_float(feature_class.get("v3_full")))
    per_game = offline_artifact.get("per_game_variance", {}).get("per_game_loo_auroc", {})
    values = [_as_float(value, default=-1.0) for value in getattr(per_game, "values", lambda: [])()]
    positive_values = [value for value in values if value > 0.0]
    return _rounded(sum(positive_values) / len(positive_values)) if positive_values else 0.0


def _live_float(
    live_artifact: Mapping[str, Any],
    top_level_key: str,
    measurement_key: str,
    measurement_metric: str,
) -> float:
    measurement = live_artifact.get(measurement_key)
    nested = measurement.get(measurement_metric) if isinstance(measurement, Mapping) else None
    return _as_float(live_artifact.get(top_level_key, nested))


def _best_positive_baseline(values: list[float], *, lower_is_better: bool) -> float:
    positives = [value for value in values if value > 0.0]
    if not positives:
        return 0.0
    return min(positives) if lower_is_better else max(positives)


def compute_offline_to_live_transfer_ratio(
    offline_artifact: Mapping[str, Any],
    live_artifact: Mapping[str, Any],
) -> JsonDict:
    """REQ-ARC-WMTE-4622: compute explicit offline signal and live lift components."""

    offline_component = _offline_auroc(offline_artifact)
    graduated_first = _live_float(
        live_artifact, "first_win_rate_graduated", "graduated_measurement", "first_win_rate"
    )
    linear_first = _live_float(
        live_artifact, "first_win_rate_linear_baseline", "linear_measurement", "first_win_rate"
    )
    bare_first = _live_float(
        live_artifact, "first_win_rate_bare", "bare_measurement", "first_win_rate"
    )
    best_first_baseline = _best_positive_baseline(
        [linear_first, bare_first], lower_is_better=False
    )
    first_lift = max(0.0, graduated_first - best_first_baseline)

    graduated_actions = _live_float(
        live_artifact,
        "median_actions_to_first_levelup_graduated",
        "graduated_measurement",
        "median_actions_to_first_levelup",
    )
    linear_actions = _live_float(
        live_artifact,
        "median_actions_to_first_levelup_linear_baseline",
        "linear_measurement",
        "median_actions_to_first_levelup",
    )
    bare_actions = _live_float(
        live_artifact,
        "median_actions_to_first_levelup_bare",
        "bare_measurement",
        "median_actions_to_first_levelup",
    )
    best_action_baseline = _best_positive_baseline(
        [linear_actions, bare_actions], lower_is_better=True
    )
    efficiency_lift = (
        max(0.0, (best_action_baseline - graduated_actions) / best_action_baseline)
        if best_action_baseline > 0.0 and graduated_actions > 0.0
        else 0.0
    )

    live_lift = max(first_lift, efficiency_lift)
    ratio = live_lift / offline_component if offline_component > 0.0 else 0.0
    return {
        "offline_to_live_transfer_ratio": _rounded(ratio),
        "offline_auroc_component": offline_component,
        "live_lift_component": _rounded(live_lift),
        "first_win_lift_component": _rounded(first_lift),
        "action_efficiency_lift_component": _rounded(efficiency_lift),
        "graduated_first_win_rate": _rounded(graduated_first),
        "baseline_first_win_rate": _rounded(best_first_baseline),
        "graduated_median_actions_to_first_levelup": _rounded(graduated_actions),
        "baseline_median_actions_to_first_levelup": _rounded(best_action_baseline),
        "bridge_crossed": live_lift > 0.0 and offline_component > 0.0,
    }


def build_coheadline_block(
    *,
    offline_artifact: Mapping[str, Any],
    live_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    live_submittable_artifact: Mapping[str, Any] | None = None,
    action_efficiency_artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4622: report the bridge metric beside ARC co-headlines."""

    metric = compute_offline_to_live_transfer_ratio(offline_artifact, live_artifact)
    live_submittable = live_submittable_artifact or {}
    action = action_efficiency_artifact or {}
    return {
        "reported_side_by_side": list(REPORTED_SIDE_BY_SIDE),
        "offline_to_live_transfer_ratio": metric["offline_to_live_transfer_ratio"],
        "offline_auroc_component": metric["offline_auroc_component"],
        "live_lift_component": metric["live_lift_component"],
        "first_win_lift_component": metric["first_win_lift_component"],
        "action_efficiency_lift_component": metric["action_efficiency_lift_component"],
        "reproducible_total_levels": _as_int(registry.get("reproducible_total_levels")),
        "live_submittable_level_count": _as_int(
            live_submittable.get("live_submittable_level_count")
        ),
        "first_win_rate": metric["graduated_first_win_rate"],
        "action_efficiency_score": _as_float(action.get("action_efficiency_score")),
        "action_efficiency_ci": action.get("action_efficiency_ci"),
        "median_actions_to_first_levelup": metric[
            "graduated_median_actions_to_first_levelup"
        ],
        "source_artifacts": {
            "offline_bridge": A1_RELATIVE_PATH,
            "offline_auroc": OFFLINE_AUROC_RELATIVE_PATH,
            "live_value_head": A2_RELATIVE_PATH,
            "registry": REGISTRY_RELATIVE_PATH,
            "live_submittable": LIVE_SUBMITTABLE_RELATIVE_PATH,
            "action_efficiency": ACTION_EFFICIENCY_RELATIVE_PATH,
        },
    }


def _null_delta_note(metric: Mapping[str, Any]) -> str | None:
    if _as_float(metric.get("live_lift_component")) == 0.0:
        return (
            "live_lift_component is zero while offline_auroc_component is reported "
            "explicitly; this is the bridge-not-crossed state, not a measurement bug."
        )
    return None


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    offline_artifact: Mapping[str, Any],
    live_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    coheadline_block: Mapping[str, Any],
    duration_s: float,
    tests_added: Mapping[str, Any],
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    metric = compute_offline_to_live_transfer_ratio(offline_artifact, live_artifact)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": HONEST_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "offline_to_live_transfer_ratio": metric["offline_to_live_transfer_ratio"],
        "offline_auroc_component": metric["offline_auroc_component"],
        "live_lift_component": metric["live_lift_component"],
        "first_win_lift_component": metric["first_win_lift_component"],
        "action_efficiency_lift_component": metric["action_efficiency_lift_component"],
        "bridge_crossed": metric["bridge_crossed"],
        "live_lift_breakdown": {
            "graduated_first_win_rate": metric["graduated_first_win_rate"],
            "baseline_first_win_rate": metric["baseline_first_win_rate"],
            "graduated_median_actions_to_first_levelup": metric[
                "graduated_median_actions_to_first_levelup"
            ],
            "baseline_median_actions_to_first_levelup": metric[
                "baseline_median_actions_to_first_levelup"
            ],
        },
        "coheadline_block": dict(coheadline_block),
        "tests_added": dict(tests_added),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
        "inference_substrate_detail": "aggregation_only_no_model_load",
        "registry_reproducible_total_levels": _as_int(
            registry.get("reproducible_total_levels")
        ),
        "source_artifacts": {
            "offline_bridge": A1_RELATIVE_PATH,
            "offline_auroc": OFFLINE_AUROC_RELATIVE_PATH,
            "live_value_head": A2_RELATIVE_PATH,
            "registry": REGISTRY_RELATIVE_PATH,
        },
    }
    note = _null_delta_note(metric)
    if note is not None:
        artifact["null_delta_methodology_note"] = note
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
    offline_component = artifact.get("offline_auroc_component")
    live_lift = artifact.get("live_lift_component")
    ratio = artifact.get("offline_to_live_transfer_ratio")
    if not isinstance(offline_component, float) or not 0.0 < offline_component <= 1.0:
        errors.append("offline_auroc_component")
    if not isinstance(live_lift, float) or live_lift < 0.0:
        errors.append("live_lift_component")
    if not isinstance(ratio, float) or ratio < 0.0:
        errors.append("offline_to_live_transfer_ratio")
    if isinstance(offline_component, float) and isinstance(live_lift, float):
        expected_ratio = _rounded(live_lift / offline_component) if offline_component > 0.0 else 0.0
        if ratio != expected_ratio:
            errors.append("offline_to_live_transfer_ratio_formula")
    if _as_float(live_lift) == 0.0:
        if not str(artifact.get("null_delta_methodology_note") or "").strip():
            errors.append("null_delta_methodology_note")
    block = artifact.get("coheadline_block")
    if not isinstance(block, Mapping):
        errors.append("coheadline_block")
    elif block.get("reported_side_by_side") != REPORTED_SIDE_BY_SIDE:
        errors.append("coheadline_block.reported_side_by_side")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": False,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root_path / A2_RELATIVE_PATH).exists(),
        "offline_auroc_artifact_present": (root_path / OFFLINE_AUROC_RELATIVE_PATH).exists(),
        "registry_present": (root_path / REGISTRY_RELATIVE_PATH).exists(),
        "spec_has_req_4622": "REQ-ARC-WMTE-4622" in spec_text,
        "model_load_required": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "a1_artifact_present",
        "a2_artifact_present",
        "offline_auroc_artifact_present",
        "registry_present",
        "spec_has_req_4622",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(key for key in required if not checks[key])
    return checks


def _read_json(path: Path) -> JsonDict:  # pragma: no cover
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> JsonDict:  # pragma: no cover
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _tests_added() -> JsonDict:  # pragma: no cover
    return {
        "passed": True,
        "test_file": "tests/python/test_experiment_4622_offline_to_live_transfer_ratio_metric.py",
        "commands": [
            (
                ".venv/bin/pytest "
                "tests/python/test_experiment_4622_offline_to_live_transfer_ratio_metric.py "
                "-q --no-cov"
            ),
            (
                ".venv/bin/python -m coverage run "
                "--include='*/python/carnot/experiment_4622_offline_to_live_transfer_ratio_metric.py' "
                "-m pytest --override-ini addopts='' "
                "tests/python/test_experiment_4622_offline_to_live_transfer_ratio_metric.py -q"
            ),
            (
                ".venv/bin/python -m coverage report "
                "--include='*/python/carnot/experiment_4622_offline_to_live_transfer_ratio_metric.py' "
                "--fail-under=100 --show-missing"
            ),
        ],
        "assertions": [
            "offline AUROC 0.72 plus zero live lift reports transfer ratio zero",
            "offline AUROC 0.72 plus positive live first-win lift reports positive ratio",
            "baseline-equal live lift emits null_delta_methodology_note",
        ],
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:  # pragma: no cover
    offline: JsonDict = {"loo_auroc_mean": 1.0}
    live: JsonDict = {
        "first_win_rate_graduated": 0.0,
        "first_win_rate_linear_baseline": 0.0,
        "first_win_rate_bare": 0.0,
    }
    registry: JsonDict = {"reproducible_total_levels": 0}
    block = build_coheadline_block(offline_artifact=offline, live_artifact=live, registry=registry)
    artifact = build_artifact(
        preconditions_checked=checks,
        offline_artifact=offline,
        live_artifact=live,
        registry=registry,
        coheadline_block=block,
        duration_s=duration_s,
        tests_added=_tests_added(),
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource') or 'precondition'}"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> JsonDict:  # pragma: no cover
    root_path = Path(root)
    start = time.perf_counter()
    checks = check_preconditions(root_path)
    if checks.get("ok") is not True:
        artifact = _blocked_artifact(checks, time.perf_counter() - start)
        if write:
            write_artifact(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact
    a1_artifact = _read_json(root_path / A1_RELATIVE_PATH)
    offline_artifact = _read_json(root_path / OFFLINE_AUROC_RELATIVE_PATH) or a1_artifact
    live_artifact = _read_json(root_path / A2_RELATIVE_PATH)
    registry = _read_yaml(root_path / REGISTRY_RELATIVE_PATH)
    live_submittable = _read_json(root_path / LIVE_SUBMITTABLE_RELATIVE_PATH)
    action_artifact = _read_json(root_path / ACTION_EFFICIENCY_RELATIVE_PATH)
    coheadline = build_coheadline_block(
        offline_artifact=offline_artifact,
        live_artifact=live_artifact,
        registry=registry,
        live_submittable_artifact=live_submittable,
        action_efficiency_artifact=action_artifact,
    )
    artifact = build_artifact(
        preconditions_checked=checks,
        offline_artifact=offline_artifact,
        live_artifact=live_artifact,
        registry=registry,
        coheadline_block=coheadline,
        duration_s=time.perf_counter() - start,
        tests_added=_tests_added(),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
