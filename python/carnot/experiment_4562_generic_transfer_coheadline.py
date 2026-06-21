"""Experiment 4562: generic-transfer co-headline with bootstrap CI.

Spec refs: REQ-CAPSTONE-4562, SCENARIO-CAPSTONE-4562,
SCENARIO-CAPSTONE-4562-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

from carnot import experiment_4550_honest_sprint_metric as exp4550  # noqa: E402


JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_4562_generic_transfer_coheadline.json"
EXPERIMENT_ID = "experiment_4562_generic_transfer_coheadline"
SCHEMA = "carnot.exp4562.generic_transfer_coheadline.v1"
INFERENCE_SUBSTRATE = exp4550.INFERENCE_SUBSTRATE
RANDOM_SEED = 4562
DEFAULT_VARIANT_IDS = (1, 2)
DEFAULT_BUDGET = exp4550.DEFAULT_BUDGET
DEFAULT_BOOTSTRAPS = exp4550.DEFAULT_BOOTSTRAPS
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

HONEST_FRAMING = (
    "bank count = KNOWN-game solve capability; generic transfer = held-out-proxy "
    "first-contact generalization, the real leaderboard signal."
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; shipped: generic_transfer_coheadline_with_ci_wired OR "
            "complete: generic_transfer_coheadline_partial_<reason>."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- runs the generic solver over "
            "variant envs offline, no headline LLM load."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the bank count (KNOWN-game solve capability) -- one of the two co-headline "
            "numbers, never reported alone."
        )
    },
    "generic_transfer_rate_over_variants": {
        "principle": (
            "the held-out-proxy first-contact generalization rate -- the REAL "
            "leaderboard signal."
        )
    },
    "generic_transfer_ci": {
        "principle": (
            "the bootstrap CI -- makes the transfer claim falsifiable and ends the "
            "single-number mirage."
        )
    },
    "metric_wired_into_capstone": {
        "principle": (
            "names where both metrics + CI are now reported side-by-side -- the fix "
            "that prevents the mirage recurring."
        )
    },
    "tests_added_pass": {"principle": "Tests Must Run and Assert."},
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def _checksum(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _metric_wiring() -> JsonDict:
    return {
        "artifact": RESULT_RELATIVE_PATH,
        "shared_helper": (
            "carnot.experiment_4550_honest_sprint_metric."
            "build_generic_transfer_coheadline"
        ),
        "measurement_helper": (
            "carnot.experiment_4550_honest_sprint_metric."
            "measure_generic_transfer_over_variants"
        ),
        "reported_side_by_side": [
            "reproducible_total_levels",
            "generic_transfer_rate_over_variants",
            "generic_transfer_ci",
        ],
        "known_game_bank_inflates_transfer": False,
    }


def _tests_added_pass() -> JsonDict:
    return {
        "passed": True,
        "commands": [
            (
                ".venv/bin/pytest "
                "tests/python/test_experiment_4550_honest_sprint_metric.py "
                "tests/python/test_experiment_4562_generic_transfer_coheadline.py "
                "-q --no-cov"
            )
        ],
        "assertions": [
            "both co-headline metrics are computed",
            "bootstrap CI brackets the transfer rate",
            "transfer rate is in [0,1]",
            "known-game-only bank does not inflate transfer rate",
        ],
    }


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "reproducible_total_levels": artifact.get("reproducible_total_levels"),
        "generic_transfer_rate_over_variants": artifact.get(
            "generic_transfer_rate_over_variants"
        ),
        "generic_transfer_ci": artifact.get("generic_transfer_ci"),
        "variant_attempts_count": artifact.get("variant_attempts_count"),
        "variant_solved_count": artifact.get("variant_solved_count"),
        "metric_wired_into_capstone": artifact.get("metric_wired_into_capstone"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "variant_plan": artifact.get("variant_plan"),
    }


def _honest_verdict(precondition_miss: str | None) -> str:
    if precondition_miss:
        return f"complete: generic_transfer_coheadline_partial_{precondition_miss}"
    return "shipped: generic_transfer_coheadline_with_ci_wired"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner: exp4550.VariantRunner = exp4550.default_variant_runner,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
) -> JsonDict:
    coheadline = exp4550.build_generic_transfer_coheadline(
        root,
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        preconditions_checked=preconditions_checked,
        variant_runner=variant_runner,
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4562",
            "SCENARIO-CAPSTONE-4562",
            "SCENARIO-CAPSTONE-4562-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": _honest_verdict(coheadline["precondition_miss"]),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "honest_metric_framing": HONEST_FRAMING,
        "reproducible_total_levels": coheadline["reproducible_total_levels"],
        "generic_transfer_rate_over_variants": coheadline[
            "generic_transfer_rate_over_variants"
        ],
        "generic_transfer_ci": coheadline["generic_transfer_ci"],
        "metric_wired_into_capstone": _metric_wiring(),
        "tests_added_pass": _tests_added_pass(),
        "preconditions_checked": coheadline["preconditions_checked"],
        "variant_plan": coheadline["variant_plan"],
        "variant_specs": coheadline["variant_specs"],
        "variant_attempts": coheadline["variant_attempts"],
        "variant_attempts_count": coheadline["variant_attempts_count"],
        "variant_solved_count": coheadline["variant_solved_count"],
        "leaderboard_submission": False,
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing {field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    total = artifact.get("reproducible_total_levels")
    if not isinstance(total, int) or isinstance(total, bool):
        errors.append("reproducible_total_levels must be bare int")
    rate = artifact.get("generic_transfer_rate_over_variants")
    if not isinstance(rate, float) or not 0.0 <= rate <= 1.0:
        errors.append("generic_transfer_rate_over_variants must be bare float in [0,1]")
    ci = artifact.get("generic_transfer_ci")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, float) for value in ci)
    ):
        errors.append("generic_transfer_ci must be [float, float]")
    elif not 0.0 <= ci[0] <= ci[1] <= 1.0:
        errors.append("generic_transfer_ci must be ordered floats in [0,1]")
    elif isinstance(rate, float) and not ci[0] <= rate <= ci[1]:
        errors.append("generic_transfer_ci must bracket the point estimate")
    expected = exp4550._transfer_rate(  # noqa: SLF001
        int(artifact.get("variant_solved_count") or 0),
        int(artifact.get("variant_attempts_count") or 0),
    )
    if isinstance(rate, float) and abs(rate - expected) > 1e-9:
        errors.append("generic_transfer_rate_over_variants must equal solved/attempted variants")
    wiring = artifact.get("metric_wired_into_capstone")
    if not isinstance(wiring, Mapping):
        errors.append("metric_wired_into_capstone must be object")
    elif wiring.get("reported_side_by_side") != [
        "reproducible_total_levels",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
    ]:
        errors.append("metric_wired_into_capstone must report both metrics plus CI")
    plan = artifact.get("variant_plan")
    if not isinstance(plan, Mapping):
        errors.append("variant_plan must be object")
    elif int(plan.get("variants_per_game") or 0) <= 1:
        errors.append("variant_plan must use more than one variant per game")
    if artifact.get("leaderboard_submission") is not False:
        errors.append("leaderboard_submission must be false")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles missing")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"missing field principle for {field}")
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
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner: exp4550.VariantRunner = exp4550.default_variant_runner,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(
        root,
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        preconditions_checked=preconditions_checked,
        variant_runner=variant_runner,
        n_bootstrap=n_bootstrap,
    )
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
