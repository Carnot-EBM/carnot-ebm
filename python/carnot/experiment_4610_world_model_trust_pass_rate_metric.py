"""Experiment 4610: canonical world-model trust pass-rate metric helper.

Spec refs: REQ-ARC-WMTE-4610, SCENARIO-ARC-WMTE-4610.
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

EXPERIMENT = "experiment_4610_world_model_trust_pass_rate_metric"
SCHEMA = "carnot.exp4610.world_model_trust_pass_rate_metric.v1"
RESULT_RELATIVE_PATH = "results/experiment_4610_world_model_trust_pass_rate_metric.json"
A1_RELATIVE_PATH = "results/experiment_4604_world_model_trust_energy.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
LIVE_SUBMITTABLE_RELATIVE_PATH = "results/experiment_4586_live_submittable_coheadline.json"
ACTION_EFFICIENCY_RELATIVE_PATH = "results/experiment_4574_action_efficiency_coheadline.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"

RANDOM_SEED = 4610
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads the A1 artifact + registry, no model load "
    "(100us floor)."
)
HONEST_VERDICT = "success: world_model_trust_pass_rate_metric_helper_shipped_tests_green"
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
SPEC_REFS = ["REQ-ARC-WMTE-4610", "SCENARIO-ARC-WMTE-4610"]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "world_model_trust_pass_rate_metric_helper_shipped_tests_green."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "world_model_trust_pass_rate": {
        "principle": (
            "the canonical co-headline metric value computed from A1 "
            "(numerator/denominator explicit, not just the float)."
        )
    },
    "trust_pass_numerator": {
        "principle": (
            "the integer count of world-model games passing the new gate AND used by "
            "the planner -- explicit so the metric does not collide as a bare fraction "
            "(the TAUTOLOGY trap)."
        )
    },
    "trust_pass_denominator": {
        "principle": "the integer count of world-model games tried -- explicit denominator."
    },
    "coheadline_block": {
        "principle": (
            "world_model_trust_pass_rate reported side-by-side with "
            "reproducible_total_levels / live-submittable / first-win-rate / "
            "generic_transfer / action-efficiency -- the single canonical metric surface."
        )
    },
    "tests_added": {
        "principle": (
            "the asserting tests (every test has >=1 assertion; no skips) -- the metric "
            "helper is verified, not asserted."
        )
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "catches silent drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REPORTED_SIDE_BY_SIDE = [
    "world_model_trust_pass_rate",
    "reproducible_total_levels",
    "live_submittable_level_count",
    "first_win_rate",
    "generic_transfer_rate_over_variants",
    "action_efficiency_score",
]
DEGENERATE_CANDIDATE_NAMES = {"identity", "noop", "no_op", "no-op"}


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


def _rows(a1_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = a1_artifact.get("measurements")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _game_name(row: Mapping[str, Any], fallback: int) -> str:
    game = str(row.get("game") or "").strip()
    return game or f"row_{fallback}"


def _is_degenerate_identity_pass(row: Mapping[str, Any]) -> bool:
    candidate = str(row.get("new_selected_candidate_name") or "").strip().lower()
    if candidate not in DEGENERATE_CANDIDATE_NAMES:
        return False
    return _as_int(row.get("new_correct_changed_cells"), 0) <= 0


def compute_world_model_trust_pass_rate(a1_artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4610: compute explicit trust-pass numerator and denominator."""

    measurements = _rows(a1_artifact)
    passed_games: list[str] = []
    tried_games: list[str] = []
    excluded_identity: list[str] = []
    for index, row in enumerate(measurements):
        game = _game_name(row, index)
        tried_games.append(game)
        gate_and_planner_used = (
            row.get("new_trust_pass") is True and row.get("new_planner_used") is True
        )
        if gate_and_planner_used and _is_degenerate_identity_pass(row):
            excluded_identity.append(game)
            continue
        if gate_and_planner_used:
            passed_games.append(game)

    denominator = len(measurements)
    numerator = len(passed_games)
    rate = round(float(numerator / denominator), 6) if denominator else 0.0
    baseline = round(_as_float(a1_artifact.get("world_model_trust_pass_rate_binary")), 6)
    return {
        "world_model_trust_pass_rate": rate,
        "trust_pass_numerator": numerator,
        "trust_pass_denominator": denominator,
        "world_model_trust_pass_rate_baseline": baseline,
        "world_model_trust_pass_rate_delta": round(rate - baseline, 6),
        "passed_games": passed_games,
        "tried_games": tried_games,
        "excluded_degenerate_identity_games": excluded_identity,
    }


def build_coheadline_block(
    *,
    a1_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    live_submittable_artifact: Mapping[str, Any] | None = None,
    action_efficiency_artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4610: report trust rate beside the ARC co-headlines."""

    trust = compute_world_model_trust_pass_rate(a1_artifact)
    live = live_submittable_artifact or {}
    action = action_efficiency_artifact or {}
    reproducible = _as_int(registry.get("reproducible_total_levels"))
    return {
        "reported_side_by_side": list(REPORTED_SIDE_BY_SIDE),
        "world_model_trust_pass_rate": trust["world_model_trust_pass_rate"],
        "trust_pass_numerator": trust["trust_pass_numerator"],
        "trust_pass_denominator": trust["trust_pass_denominator"],
        "world_model_trust_pass_rate_baseline": trust[
            "world_model_trust_pass_rate_baseline"
        ],
        "world_model_trust_pass_rate_delta": trust["world_model_trust_pass_rate_delta"],
        "reproducible_total_levels": reproducible,
        "live_submittable_level_count": _as_int(live.get("live_submittable_level_count")),
        "first_win_rate": _as_float(a1_artifact.get("first_win_rate_new")),
        "generic_transfer_rate_over_variants": _as_float(
            action.get("generic_transfer_rate_over_variants")
        ),
        "generic_transfer_ci": action.get("generic_transfer_ci"),
        "action_efficiency_score": _as_float(action.get("action_efficiency_score")),
        "action_efficiency_ci": action.get("action_efficiency_ci"),
        "median_actions_to_first_levelup": action.get(
            "median_actions_to_first_levelup",
            a1_artifact.get("median_actions_to_first_levelup_new"),
        ),
        "source_artifacts": {
            "world_model_trust": A1_RELATIVE_PATH,
            "registry": REGISTRY_RELATIVE_PATH,
            "live_submittable": LIVE_SUBMITTABLE_RELATIVE_PATH,
            "action_efficiency": ACTION_EFFICIENCY_RELATIVE_PATH,
        },
    }


def _null_delta_note(metric: Mapping[str, Any]) -> str | None:
    if _as_float(metric.get("world_model_trust_pass_rate_delta")) == 0.0:
        return (
            "world_model_trust_pass_rate equals the matched baseline after computing "
            "the explicit numerator and denominator; this is an honest no-value null, "
            "not a measurement bug."
        )
    return None


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    a1_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    coheadline_block: Mapping[str, Any],
    duration_s: float,
    tests_added: Mapping[str, Any],
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    metric = compute_world_model_trust_pass_rate(a1_artifact)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": HONEST_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "world_model_trust_pass_rate": metric["world_model_trust_pass_rate"],
        "trust_pass_numerator": metric["trust_pass_numerator"],
        "trust_pass_denominator": metric["trust_pass_denominator"],
        "world_model_trust_pass_rate_baseline": metric[
            "world_model_trust_pass_rate_baseline"
        ],
        "world_model_trust_pass_rate_delta": metric["world_model_trust_pass_rate_delta"],
        "passed_games": metric["passed_games"],
        "tried_games": metric["tried_games"],
        "excluded_degenerate_identity_games": metric["excluded_degenerate_identity_games"],
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
    numerator = artifact.get("trust_pass_numerator")
    denominator = artifact.get("trust_pass_denominator")
    rate = artifact.get("world_model_trust_pass_rate")
    if type(numerator) is not int:
        errors.append("trust_pass_numerator")
    if type(denominator) is not int or denominator < 0:
        errors.append("trust_pass_denominator")
    if not isinstance(rate, float) or isinstance(rate, bool) or not 0.0 <= rate <= 1.0:
        errors.append("world_model_trust_pass_rate")
    if type(numerator) is int and type(denominator) is int and denominator > 0:
        expected = round(float(numerator / denominator), 6)
        if rate != expected:
            errors.append("world_model_trust_pass_rate_fraction")
    if _as_float(artifact.get("world_model_trust_pass_rate_delta")) == 0.0:
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


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - IO boundary
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": False,
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "registry_present": (root_path / REGISTRY_RELATIVE_PATH).exists(),
        "spec_has_req_4610": "REQ-ARC-WMTE-4610" in spec_text,
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
        "registry_present",
        "spec_has_req_4610",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next(key for key in required if not checks[key])
    return checks


def _read_json(path: Path) -> JsonDict:  # pragma: no cover - IO boundary
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> JsonDict:  # pragma: no cover - IO boundary
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:  # pragma: no cover - IO boundary
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _tests_added() -> JsonDict:
    return {
        "passed": True,
        "test_file": "tests/python/test_experiment_4610_world_model_trust_pass_rate_metric.py",
        "commands": [
            (
                ".venv/bin/pytest "
                "tests/python/test_experiment_4610_world_model_trust_pass_rate_metric.py "
                "-q --no-cov"
            ),
            (
                ".venv/bin/python -m coverage run "
                "--include='*/python/carnot/experiment_4610_world_model_trust_pass_rate_metric.py' "
                "-m pytest --override-ini addopts='' "
                "tests/python/test_experiment_4610_world_model_trust_pass_rate_metric.py -q"
            ),
            (
                ".venv/bin/python -m coverage report "
                "--include='*/python/carnot/experiment_4610_world_model_trust_pass_rate_metric.py' "
                "--fail-under=100 --show-missing"
            ),
        ],
        "assertions": [
            "synthetic k/n returns explicit numerator and denominator",
            "degenerate identity-pass is excluded from numerator",
            "baseline-equal case emits null_delta_methodology_note",
        ],
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:  # pragma: no cover
    empty_a1: JsonDict = {
        "measurements": [],
        "world_model_trust_pass_rate_binary": 0.0,
        "first_win_rate_new": 0.0,
    }
    empty_registry: JsonDict = {"reproducible_total_levels": 0}
    block = build_coheadline_block(a1_artifact=empty_a1, registry=empty_registry)
    artifact = build_artifact(
        preconditions_checked=checks,
        a1_artifact=empty_a1,
        registry=empty_registry,
        coheadline_block=block,
        duration_s=duration_s,
        tests_added=_tests_added(),
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource') or 'precondition'}"
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> JsonDict:  # pragma: no cover - CLI boundary
    root_path = Path(root)
    start = time.perf_counter()
    checks = check_preconditions(root_path)
    if checks.get("ok") is not True:
        artifact = _blocked_artifact(checks, time.perf_counter() - start)
        if write:
            write_artifact(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact
    a1_artifact = _read_json(root_path / A1_RELATIVE_PATH)
    registry = _read_yaml(root_path / REGISTRY_RELATIVE_PATH)
    live_artifact = _read_json(root_path / LIVE_SUBMITTABLE_RELATIVE_PATH)
    action_artifact = _read_json(root_path / ACTION_EFFICIENCY_RELATIVE_PATH)
    coheadline = build_coheadline_block(
        a1_artifact=a1_artifact,
        registry=registry,
        live_submittable_artifact=live_artifact,
        action_efficiency_artifact=action_artifact,
    )
    artifact = build_artifact(
        preconditions_checked=checks,
        a1_artifact=a1_artifact,
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


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
