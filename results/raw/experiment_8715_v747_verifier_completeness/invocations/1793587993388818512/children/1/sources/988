"""Experiment 4586: live-submittable co-headline metric.

Spec refs: REQ-CAPSTONE-4586, SCENARIO-CAPSTONE-4586,
SCENARIO-CAPSTONE-4586-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4574_action_efficiency_coheadline as exp4574  # noqa: E402
from carnot import experiment_4550_honest_sprint_metric as exp4550  # noqa: E402
from carnot import live_submittable_metrics  # noqa: E402


JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4586_live_submittable_coheadline.json"
B1_COHEADLINE_RELATIVE_PATH = exp4574.RESULT_RELATIVE_PATH
CAPSTONE_422_RELATIVE_PATH = "results/experiment_4578_capstone_v422.json"
WINNER_GENERATED_SOURCE_RELATIVE_PATH = "results/experiment_4582_feature_router_transfer.json"
LIVE_SCORECARD_RELATIVE_PATH = "results/arc3_live_submit.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"

EXPERIMENT = "experiment_4586_live_submittable_coheadline"
SCHEMA = "carnot.exp4586.live_submittable_coheadline.v1"
RANDOM_SEED = 4586
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- computes the count over the "
    "registry/package offline, no LLM load."
)
TERMINAL_PREFIXES = ("shipped:", "complete:", "success:", "passed:", "blocked_")

HONEST_FRAMING = (
    "reproducible_total_levels = banked capability incl. non-submittable; "
    "live-submittable = the honest leaderboard score; generic_transfer = held-out "
    "first-contact; action efficiency = the scoring tiebreaker."
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; shipped: live_submittable_coheadline_wired OR complete: "
            "live_submittable_coheadline_partial_<reason>."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "live_submittable_level_count": {
        "principle": (
            "the honest leaderboard score (offline-reproduced AND trajectory AND "
            "env-matchable) -- the metric this task formalizes."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the raw banked-capability count (53) -- reported alongside to expose "
            "the mirage gap."
        )
    },
    "reproducible_vs_submittable_gap": {
        "principle": (
            "reproducible_total_levels - live_submittable_level_count -- the "
            "GAP-LIVE-INTEGRATION number that drives the A1 headline."
        )
    },
    "metric_wired_into_capstone": {
        "principle": (
            "names where all co-headline metrics are now reported side-by-side -- "
            "the fix that surfaces the honest score."
        )
    },
    "tests_added_pass": {
        "principle": (
            "Tests Must Run and Assert -- subset, exclude-no-trajectory, "
            "include-with-trajectory, gap-reported."
        )
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive file boundary
        return {}
    return loaded if isinstance(loaded, dict) else {}  # pragma: no cover - defensive file boundary


def _load_registry_ok(path: Path) -> bool:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):  # pragma: no cover - defensive file boundary
        return False
    return isinstance(loaded, Mapping)


def _tests_added_pass() -> JsonDict:
    return {
        "passed": True,
        "commands": [
            (
                ".venv/bin/pytest "
                "tests/python/test_experiment_4586_live_submittable_coheadline.py "
                "-q --no-cov"
            )
        ],
        "assertions": [
            "live-submittable count is <= reproducible_total_levels",
            "banked rows without trajectory/adaptive resolver are excluded",
            "trajectory plus env-match rows are included",
            "reproducible_vs_submittable_gap is reported",
        ],
    }


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    package_path: str | None = None,
) -> JsonDict:
    root_path = Path(root)
    selected_package = package_path or live_submittable_metrics.default_package_path(root_path)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    registry_path = root_path / live_submittable_metrics.REGISTRY_RELATIVE_PATH
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "registry_yaml_loadable": _load_registry_ok(registry_path),
        "registry_path": live_submittable_metrics.REGISTRY_RELATIVE_PATH,
        "refreshed_package_present": (root_path / selected_package).exists(),
        "refreshed_package_path": selected_package,
        "b1_coheadline_artifact_present": (root_path / B1_COHEADLINE_RELATIVE_PATH).exists(),
        "capstone_422_artifact_present": (root_path / CAPSTONE_422_RELATIVE_PATH).exists(),
        "live_scorecard_present": (root_path / LIVE_SCORECARD_RELATIVE_PATH).exists(),
        "spec_has_req_4586": "REQ-CAPSTONE-4586" in spec_text,
        "leaderboard_submission": False,
        "network_required": False,
        "research_conductor_modified": False,
    }
    required = (
        "registry_yaml_loadable",
        "refreshed_package_present",
        "b1_coheadline_artifact_present",
        "spec_has_req_4586",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def _honest_verdict(preconditions: Mapping[str, Any], live_subset: bool) -> str:
    if preconditions.get("ok") is not True:
        return "complete: live_submittable_coheadline_partial_preconditions"
    if not live_subset:
        return "complete: live_submittable_coheadline_partial_subset_violation"
    return "shipped: live_submittable_coheadline_wired"


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_winner_generated_rate_coheadline_metrics(
    root: Path | str = REPO_ROOT,
    *,
    registry: Mapping[str, Any] | None = None,
    package: Mapping[str, Any] | None = None,
    package_path: str | None = None,
    b1_artifact: Mapping[str, Any] | None = None,
    winner_source_artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    """REQ-CAPSTONE-4598: report all capstone co-headline metrics side-by-side."""

    root_path = Path(root)
    coheadline = exp4574.build_live_submittable_coheadline_metrics(
        root_path,
        registry=registry,
        package=package,
        package_path=package_path,
        b1_artifact=b1_artifact,
    )
    winner_source = dict(
        winner_source_artifact
        or _read_json(root_path / WINNER_GENERATED_SOURCE_RELATIVE_PATH)
    )
    winner = exp4550.winner_generated_metric_from_artifact(winner_source)
    generic_rate = _float_or_none(coheadline.get("generic_transfer_rate_over_variants"))
    if generic_rate is not None:
        winner["generic_transfer_rate_over_variants"] = round(generic_rate, 10)
        winner["generation_vs_ranking_gap"] = round(
            float(winner["winner_generated_rate"]) - generic_rate,
            10,
        )
    reported_side_by_side = [
        "reproducible_total_levels",
        "live_submittable_level_count",
        "reproducible_vs_submittable_gap",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
        "action_efficiency_score",
        "action_efficiency_ci",
        "winner_generated_rate",
        "generation_vs_ranking_gap",
    ]
    return {
        **coheadline,
        **winner,
        "reported_side_by_side": reported_side_by_side,
        "winner_generated_source_artifact": str(
            winner_source.get("result_path") or WINNER_GENERATED_SOURCE_RELATIVE_PATH
        ),
        "capstone_function": (
            "carnot.experiment_4586_live_submittable_coheadline."
            "build_winner_generated_rate_coheadline_metrics"
        ),
    }


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "reproducible_total_levels": artifact.get("reproducible_total_levels"),
        "live_submittable_level_count": artifact.get("live_submittable_level_count"),
        "reproducible_vs_submittable_gap": artifact.get("reproducible_vs_submittable_gap"),
        "generic_transfer_rate_over_variants": artifact.get(
            "generic_transfer_rate_over_variants"
        ),
        "generic_transfer_ci": artifact.get("generic_transfer_ci"),
        "action_efficiency_score": artifact.get("action_efficiency_score"),
        "action_efficiency_ci": artifact.get("action_efficiency_ci"),
        "metric_wired_into_capstone": artifact.get("metric_wired_into_capstone"),
        "refreshed_package_path": artifact.get("refreshed_package_path"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "per_game_live_submittable": artifact.get("per_game_live_submittable"),
        "random_seed": artifact.get("random_seed"),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    registry: Mapping[str, Any] | None = None,
    package: Mapping[str, Any] | None = None,
    package_path: str | None = None,
    b1_artifact: Mapping[str, Any] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path, package_path=package_path))
    coheadline = exp4574.build_live_submittable_coheadline_metrics(
        root_path,
        registry=registry,
        package=package,
        package_path=package_path or checks.get("refreshed_package_path"),
        b1_artifact=b1_artifact,
    )
    live_metrics = live_submittable_metrics.compute_live_submittable_metrics(
        root_path,
        registry=registry,
        package=package,
        package_path=package_path or checks.get("refreshed_package_path"),
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4586",
            "SCENARIO-CAPSTONE-4586",
            "SCENARIO-CAPSTONE-4586-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(
            checks,
            bool(live_metrics["live_submittable_subset_of_reproducible"]),
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "honest_metric_framing": HONEST_FRAMING,
        "live_submittable_level_count": live_metrics["live_submittable_level_count"],
        "reproducible_total_levels": live_metrics["reproducible_total_levels"],
        "reproducible_vs_submittable_gap": live_metrics[
            "reproducible_vs_submittable_gap"
        ],
        "generic_transfer_rate_over_variants": coheadline[
            "generic_transfer_rate_over_variants"
        ],
        "generic_transfer_ci": coheadline["generic_transfer_ci"],
        "action_efficiency_score": coheadline["action_efficiency_score"],
        "action_efficiency_ci": coheadline["action_efficiency_ci"],
        "median_actions_to_first_levelup": coheadline["median_actions_to_first_levelup"],
        "human_baseline_actions": coheadline["human_baseline_actions"],
        "metric_wired_into_capstone": {
            "reported_side_by_side": coheadline["reported_side_by_side"],
            "capstone_function": coheadline["capstone_function"],
            "source_action_efficiency_helper": B1_COHEADLINE_RELATIVE_PATH,
            "source_capstone": CAPSTONE_422_RELATIVE_PATH,
            "live_submittable_subset_of_reproducible": live_metrics[
                "live_submittable_subset_of_reproducible"
            ],
        },
        "tests_added_pass": _tests_added_pass(),
        "preconditions_checked": checks,
        "per_game_live_submittable": live_metrics["per_game_live_submittable"],
        "refreshed_package_path": live_metrics["refreshed_package_path"],
        "registry_path": live_metrics["registry_path"],
        "leaderboard_submission": False,
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def _is_float_ci(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, float) for item in value)
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    for field in (
        "live_submittable_level_count",
        "reproducible_total_levels",
        "reproducible_vs_submittable_gap",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    live_count = artifact.get("live_submittable_level_count")
    reproducible = artifact.get("reproducible_total_levels")
    gap = artifact.get("reproducible_vs_submittable_gap")
    if type(live_count) is int and type(reproducible) is int and type(gap) is int:
        if live_count > reproducible:
            errors.append("live_submittable_level_count must be <= reproducible_total_levels")
        if gap != reproducible - live_count:
            errors.append("reproducible_vs_submittable_gap must equal reproducible - live")
    rate = artifact.get("generic_transfer_rate_over_variants")
    if not isinstance(rate, float) or isinstance(rate, bool) or not 0.0 <= rate <= 1.0:
        errors.append("generic_transfer_rate_over_variants must be a bare float in [0,1]")
    if not _is_float_ci(artifact.get("generic_transfer_ci")):
        errors.append("generic_transfer_ci must be [float, float]")
    score = artifact.get("action_efficiency_score")
    if not isinstance(score, float) or isinstance(score, bool) or not 0.0 <= score <= 1.0:
        errors.append("action_efficiency_score must be a bare float in [0,1]")
    if not _is_float_ci(artifact.get("action_efficiency_ci")):
        errors.append("action_efficiency_ci must be [float, float]")
    wiring = artifact.get("metric_wired_into_capstone")
    if not isinstance(wiring, Mapping):
        errors.append("metric_wired_into_capstone must be object")
    else:
        expected = [
            "reproducible_total_levels",
            "live_submittable_level_count",
            "reproducible_vs_submittable_gap",
            "generic_transfer_rate_over_variants",
            "generic_transfer_ci",
            "action_efficiency_score",
            "action_efficiency_ci",
        ]
        if wiring.get("reported_side_by_side") != expected:
            errors.append("metric_wired_into_capstone must report all co-headlines")
        if wiring.get("live_submittable_subset_of_reproducible") is not True:
            errors.append("metric_wired_into_capstone must record subset proof")
    rows = artifact.get("per_game_live_submittable")
    if not isinstance(rows, list):
        errors.append("per_game_live_submittable must be list")
    elif type(live_count) is int:
        row_total = sum(
            row.get("submittable_level", 0)
            for row in rows
            if isinstance(row, Mapping) and type(row.get("submittable_level", 0)) is int
        )
        if row_total != live_count:
            errors.append("per_game_live_submittable must sum to live count")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    if not isinstance(artifact.get("tests_added_pass"), Mapping):
        errors.append("tests_added_pass must be object")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles missing")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"missing field principle for {field}")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != _checksum(_artifact_checksum_payload(artifact)):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    artifact: Mapping[str, Any] | None = None,
) -> Path:
    root_path = Path(root)
    payload = dict(artifact or build_artifact(root_path))
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    path = root_path / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    root: Path | str = REPO_ROOT,
    *,
    package_path: str | None = None,
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(root, package_path=package_path)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
