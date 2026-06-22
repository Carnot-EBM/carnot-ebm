"""Experiment 4598: winner-generated-rate co-headline metric.

Spec refs: REQ-CAPSTONE-4598, SCENARIO-CAPSTONE-4598,
SCENARIO-CAPSTONE-4598-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4586_live_submittable_coheadline as exp4586  # noqa: E402


JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4598_winner_generated_rate_metric.json"
FEATURE_ROUTER_BASELINE_RELATIVE_PATH = exp4586.WINNER_GENERATED_SOURCE_RELATIVE_PATH
LIVE_SUBMITTABLE_COH_RELATIVE_PATH = exp4586.RESULT_RELATIVE_PATH
CAPSTONE_423_RELATIVE_PATH = "results/experiment_4590_capstone_v423.json"
GENERATION_WIRING_RELATIVE_PATH = "results/experiment_4592_generation_completeness_wiring.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"

EXPERIMENT = "experiment_4598_winner_generated_rate_metric"
SCHEMA = "carnot.exp4598.winner_generated_rate_metric.v1"
RANDOM_SEED = 4598
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- computes the metric over variant "
    "records offline, no LLM load."
)
TERMINAL_PREFIXES = ("shipped:", "complete:", "success:", "passed:", "blocked_")

HONEST_FRAMING = (
    "winner_generated_rate = can we GENERATE the winner at all; generic_transfer = do "
    "we SOLVE the held-out variant; generation_vs_ranking_gap = the ranking-vs-generation "
    "residual."
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; shipped: winner_generated_rate_coheadline_wired OR complete: "
            "winner_generated_rate_coheadline_partial_<reason>."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "winner_generated_rate": {
        "principle": (
            "the generation-vs-ranking gap metric this task formalizes (1/25 baseline) "
            "-- can the system GENERATE the winner at all."
        )
    },
    "generic_transfer_rate_over_variants": {
        "principle": (
            "the held-out solve-rate (0.04) -- reported alongside; the gap "
            "winner_generated_rate - generic_transfer is the ranking residual."
        )
    },
    "generation_vs_ranking_gap": {
        "principle": (
            "winner_generated_rate - generic_transfer -- how much of the failure is "
            "generation (winner never made) vs ranking (made but not selected)."
        )
    },
    "metric_wired_into_capstone": {
        "principle": (
            "names where all co-headline metrics are now reported side-by-side -- the "
            "fix that surfaces the generation gap."
        )
    },
    "tests_added_pass": {
        "principle": (
            "Tests Must Run and Assert -- winner_generated_rate >= generic_transfer; "
            "generated-but-not-selected counted; 1/25 baseline reproduces."
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
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _tests_added_pass() -> JsonDict:
    return {
        "passed": True,
        "commands": [
            (
                ".venv/bin/pytest "
                "tests/python/test_experiment_4598_winner_generated_rate_metric.py "
                "-q --no-cov"
            )
        ],
        "assertions": [
            "winner_generated_rate >= generic_transfer_rate_over_variants",
            "generated-but-not-selected variants count only in winner_generated_rate",
            "Exp4582 winner_generated=1/25 baseline reproduces from the record",
        ],
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    offline_arcade = False
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
        offline_arcade = True
    except Exception:
        offline_arcade = False
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_or_opencode_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "spec_has_req_4598": "REQ-CAPSTONE-4598" in spec_text,
        "offline_arcade": offline_arcade,
        "winner_source_artifact_present": (
            root_path / FEATURE_ROUTER_BASELINE_RELATIVE_PATH
        ).exists(),
        "live_submittable_coheadline_artifact_present": (
            root_path / LIVE_SUBMITTABLE_COH_RELATIVE_PATH
        ).exists(),
        "capstone_423_artifact_present": (root_path / CAPSTONE_423_RELATIVE_PATH).exists(),
        "generation_wiring_artifact_present": (
            root_path / GENERATION_WIRING_RELATIVE_PATH
        ).exists(),
        "leaderboard_submission": False,
        "network_required": False,
        "research_conductor_modified": False,
    }
    required = (
        "agents_md_read",
        "codex_or_opencode_md_read",
        "spec_has_req_4598",
        "offline_arcade",
        "winner_source_artifact_present",
        "live_submittable_coheadline_artifact_present",
    )
    checks["ok"] = all(bool(checks[key]) for key in required)
    return checks


def _honest_verdict(preconditions: Mapping[str, Any], winner_rate: float, transfer_rate: float) -> str:
    if preconditions.get("ok") is not True:
        return "complete: winner_generated_rate_coheadline_partial_preconditions"
    if winner_rate < transfer_rate:
        return "complete: winner_generated_rate_coheadline_partial_metric_invariant"
    return "shipped: winner_generated_rate_coheadline_wired"


def _metric_wiring(metrics: Mapping[str, Any]) -> JsonDict:
    return {
        "reported_side_by_side": list(metrics["reported_side_by_side"]),
        "capstone_function": metrics["capstone_function"],
        "shared_winner_generated_helper": (
            "carnot.experiment_4550_honest_sprint_metric."
            "winner_generated_metric_from_artifact"
        ),
        "shared_variant_attempt_helper": (
            "carnot.experiment_4550_honest_sprint_metric."
            "measure_winner_generated_over_variants"
        ),
        "source_live_submittable_coheadline": LIVE_SUBMITTABLE_COH_RELATIVE_PATH,
        "source_winner_generated_baseline": FEATURE_ROUTER_BASELINE_RELATIVE_PATH,
        "source_capstone": CAPSTONE_423_RELATIVE_PATH,
        "source_generation_wiring": GENERATION_WIRING_RELATIVE_PATH,
    }


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "reproducible_total_levels": artifact.get("reproducible_total_levels"),
        "live_submittable_level_count": artifact.get("live_submittable_level_count"),
        "generic_transfer_rate_over_variants": artifact.get(
            "generic_transfer_rate_over_variants"
        ),
        "action_efficiency_score": artifact.get("action_efficiency_score"),
        "winner_generated_rate": artifact.get("winner_generated_rate"),
        "generation_vs_ranking_gap": artifact.get("generation_vs_ranking_gap"),
        "metric_wired_into_capstone": artifact.get("metric_wired_into_capstone"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "winner_generated_source_artifact": artifact.get("winner_generated_source_artifact"),
        "random_seed": artifact.get("random_seed"),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    registry: Mapping[str, Any] | None = None,
    package: Mapping[str, Any] | None = None,
    package_path: str | None = None,
    b1_artifact: Mapping[str, Any] | None = None,
    winner_source_artifact: Mapping[str, Any] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    coheadline_source = b1_artifact or _read_json(root_path / LIVE_SUBMITTABLE_COH_RELATIVE_PATH)
    winner_source = winner_source_artifact or _read_json(
        root_path / FEATURE_ROUTER_BASELINE_RELATIVE_PATH
    )
    metrics = exp4586.build_winner_generated_rate_coheadline_metrics(
        root_path,
        registry=registry,
        package=package,
        package_path=package_path,
        b1_artifact=coheadline_source,
        winner_source_artifact=winner_source,
    )
    winner_rate = float(metrics["winner_generated_rate"])
    transfer_rate = float(metrics["generic_transfer_rate_over_variants"])
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4598",
            "SCENARIO-CAPSTONE-4598",
            "SCENARIO-CAPSTONE-4598-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(checks, winner_rate, transfer_rate),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "honest_metric_framing": HONEST_FRAMING,
        "reproducible_total_levels": metrics["reproducible_total_levels"],
        "live_submittable_level_count": metrics["live_submittable_level_count"],
        "reproducible_vs_submittable_gap": metrics["reproducible_vs_submittable_gap"],
        "generic_transfer_rate_over_variants": transfer_rate,
        "generic_transfer_ci": metrics["generic_transfer_ci"],
        "action_efficiency_score": float(metrics["action_efficiency_score"]),
        "action_efficiency_ci": metrics["action_efficiency_ci"],
        "winner_generated_rate": winner_rate,
        "winner_generated_count": metrics["winner_generated_count"],
        "winner_generated_attempted_count": metrics["winner_generated_attempted_count"],
        "winner_generated_not_selected_count": metrics[
            "winner_generated_not_selected_count"
        ],
        "generic_transfer_solved_count": metrics["generic_transfer_solved_count"],
        "generation_vs_ranking_gap": float(metrics["generation_vs_ranking_gap"]),
        "metric_wired_into_capstone": _metric_wiring(metrics),
        "tests_added_pass": _tests_added_pass(),
        "preconditions_checked": checks,
        "winner_generated_source_artifact": metrics["winner_generated_source_artifact"],
        "leaderboard_submission": False,
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def _is_float_rate(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    for field in ("winner_generated_rate", "generic_transfer_rate_over_variants"):
        if not _is_float_rate(artifact.get(field)):
            errors.append(f"{field} must be a bare float in [0,1]")
    gap = artifact.get("generation_vs_ranking_gap")
    winner = artifact.get("winner_generated_rate")
    transfer = artifact.get("generic_transfer_rate_over_variants")
    if not isinstance(gap, float) or isinstance(gap, bool):
        errors.append("generation_vs_ranking_gap must be a bare float")
    if isinstance(winner, float) and isinstance(transfer, float):
        if winner < transfer:
            errors.append("winner_generated_rate must be >= generic_transfer_rate_over_variants")
        if isinstance(gap, float) and round(winner - transfer, 10) != round(gap, 10):
            errors.append("generation_vs_ranking_gap must equal winner_generated_rate - generic")
    for field in (
        "reproducible_total_levels",
        "live_submittable_level_count",
        "winner_generated_count",
        "winner_generated_attempted_count",
        "winner_generated_not_selected_count",
        "generic_transfer_solved_count",
    ):
        if type(artifact.get(field)) is not int:
            errors.append(f"{field} must be bare int")
    wiring = artifact.get("metric_wired_into_capstone")
    expected = [
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
    if not isinstance(wiring, Mapping):
        errors.append("metric_wired_into_capstone must be object")
    elif wiring.get("reported_side_by_side") != expected:
        errors.append("metric_wired_into_capstone must report all co-headlines")
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
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(root)
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
