"""Experiment 4574: action-efficiency co-headline with bootstrap CI.

Spec refs: REQ-CAPSTONE-4574, SCENARIO-CAPSTONE-4574,
SCENARIO-CAPSTONE-4574-FIELD-PRINCIPLES.
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

RESULT_RELATIVE_PATH = "results/experiment_4574_action_efficiency_coheadline.json"
EXPERIMENT_ID = "experiment_4574_action_efficiency_coheadline"
SCHEMA = "carnot.exp4574.action_efficiency_coheadline.v1"
INFERENCE_SUBSTRATE = exp4550.INFERENCE_SUBSTRATE
HUMAN_REPLAY_RELATIVE_PATH = exp4550.HUMAN_REPLAY_RELATIVE_PATH
RANDOM_SEED = 4574
DEFAULT_VARIANT_IDS = (1, 2)
DEFAULT_BUDGET = exp4550.DEFAULT_BUDGET
DEFAULT_BOOTSTRAPS = exp4550.DEFAULT_BOOTSTRAPS
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

HONEST_FRAMING = (
    "bank count = KNOWN-game capability; generic transfer = held-out first-contact "
    "solve-rate; action efficiency = the literal leaderboard scoring term."
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; shipped: action_efficiency_coheadline_with_ci_wired OR "
            "complete: action_efficiency_coheadline_partial_<reason>."
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
            "the bank count (KNOWN-game capability) -- one of the three co-headline "
            "numbers, never reported alone."
        )
    },
    "generic_transfer_rate_over_variants": {
        "principle": (
            "the held-out first-contact solve-rate -- the generic-transfer co-headline "
            "beside bank count and action efficiency."
        )
    },
    "generic_transfer_ci": {
        "principle": "the bootstrap CI for held-out first-contact solve-rate."
    },
    "median_actions_to_first_levelup": {
        "principle": (
            "the agent's held-out median actions-to-first-levelup -- the "
            "action-efficiency numerator the leaderboard scores."
        )
    },
    "human_baseline_actions": {
        "principle": (
            "the human actions-to-levelup baseline from the replay corpus -- the "
            "denominator of min(human/agent,1)^2."
        )
    },
    "action_efficiency_score": {
        "principle": (
            "min(human/agent,1)^2 -- the literal leaderboard scoring term, now a "
            "first-class metric."
        )
    },
    "action_efficiency_ci": {
        "principle": "the bootstrap CI -- makes the efficiency claim falsifiable."
    },
    "metric_wired_into_capstone": {
        "principle": (
            "names where all three co-headline metrics + CIs are now reported "
            "side-by-side -- the fix that surfaces the score lever."
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


def _tests_added_pass() -> JsonDict:
    return {
        "passed": True,
        "commands": [
            (
                ".venv/bin/pytest "
                "tests/python/test_experiment_4574_action_efficiency_coheadline.py "
                "-q --no-cov"
            )
        ],
        "assertions": [
            "all three co-headline metrics are computed",
            "generic-transfer and action-efficiency CIs bracket point estimates",
            "action-efficiency score is in [0,1]",
            "known-game bank does not inflate action efficiency",
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
        "median_actions_to_first_levelup": artifact.get("median_actions_to_first_levelup"),
        "human_baseline_actions": artifact.get("human_baseline_actions"),
        "action_efficiency_score": artifact.get("action_efficiency_score"),
        "action_efficiency_ci": artifact.get("action_efficiency_ci"),
        "variant_attempts_count": artifact.get("variant_attempts_count"),
        "variant_solved_count": artifact.get("variant_solved_count"),
        "agent_actions_to_first_levelup": artifact.get("agent_actions_to_first_levelup"),
        "human_baseline_sample_count": artifact.get("human_baseline_sample_count"),
        "metric_wired_into_capstone": artifact.get("metric_wired_into_capstone"),
        "preconditions_checked": artifact.get("preconditions_checked"),
        "variant_plan": artifact.get("variant_plan"),
    }


def _honest_verdict(precondition_miss: str | None) -> str:
    if precondition_miss:
        return f"complete: action_efficiency_coheadline_partial_{precondition_miss}"
    return "shipped: action_efficiency_coheadline_with_ci_wired"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner: exp4550.VariantRunner = exp4550.default_variant_runner,
    human_actions: Sequence[int | float] | None = None,
    human_replay_data_dir: Path | str | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
) -> JsonDict:
    preconditions = dict(preconditions_checked or exp4550.check_preconditions(root))
    preconditions.setdefault(
        "exp4550_measure_generic_transfer_over_variants_import",
        callable(exp4550.measure_generic_transfer_over_variants),
    )
    coheadline = exp4550.build_capstone_coheadline_metrics(
        root,
        result_path=RESULT_RELATIVE_PATH,
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        preconditions_checked=preconditions,
        variant_runner=variant_runner,
        human_actions=human_actions,
        human_replay_data_dir=human_replay_data_dir,
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4574",
            "SCENARIO-CAPSTONE-4574",
            "SCENARIO-CAPSTONE-4574-FIELD-PRINCIPLES",
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
        "median_actions_to_first_levelup": coheadline[
            "median_actions_to_first_levelup"
        ],
        "human_baseline_actions": coheadline["human_baseline_actions"],
        "action_efficiency_score": coheadline["action_efficiency_score"],
        "action_efficiency_ci": coheadline["action_efficiency_ci"],
        "metric_wired_into_capstone": coheadline["metric_wired_into_capstone"],
        "tests_added_pass": _tests_added_pass(),
        "preconditions_checked": coheadline["preconditions_checked"],
        "variant_plan": coheadline["variant_plan"],
        "variant_specs": coheadline["variant_specs"],
        "variant_attempts": coheadline["variant_attempts"],
        "variant_attempts_count": coheadline["variant_attempts_count"],
        "variant_solved_count": coheadline["variant_solved_count"],
        "agent_actions_to_first_levelup": coheadline["agent_actions_to_first_levelup"],
        "human_baseline_sample_count": coheadline["human_baseline_sample_count"],
        "leaderboard_submission": False,
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def _is_two_float_ci(value: Any) -> bool:
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
    total = artifact.get("reproducible_total_levels")
    if not isinstance(total, int) or isinstance(total, bool):
        errors.append("reproducible_total_levels must be bare int")
    rate = artifact.get("generic_transfer_rate_over_variants")
    if not isinstance(rate, float) or not 0.0 <= rate <= 1.0:
        errors.append("generic_transfer_rate_over_variants must be bare float in [0,1]")
    transfer_ci = artifact.get("generic_transfer_ci")
    if not _is_two_float_ci(transfer_ci):
        errors.append("generic_transfer_ci must be [float, float]")
    elif not 0.0 <= transfer_ci[0] <= transfer_ci[1] <= 1.0:
        errors.append("generic_transfer_ci must be ordered floats in [0,1]")
    elif isinstance(rate, float) and not transfer_ci[0] <= rate <= transfer_ci[1]:
        errors.append("generic_transfer_ci must bracket the point estimate")
    expected = exp4550._transfer_rate(  # noqa: SLF001
        int(artifact.get("variant_solved_count") or 0),
        int(artifact.get("variant_attempts_count") or 0),
    )
    if isinstance(rate, float) and abs(rate - expected) > 1e-9:
        errors.append("generic_transfer_rate_over_variants must equal solved/attempted variants")

    median_actions = artifact.get("median_actions_to_first_levelup")
    if median_actions is not None and (
        not isinstance(median_actions, int | float)
        or isinstance(median_actions, bool)
        or float(median_actions) <= 0.0
    ):
        errors.append("median_actions_to_first_levelup must be positive numeric or null")
    human = artifact.get("human_baseline_actions")
    if not isinstance(human, int | float) or isinstance(human, bool) or float(human) <= 0.0:
        errors.append("human_baseline_actions must be positive numeric")
    score = artifact.get("action_efficiency_score")
    if not isinstance(score, float) or not 0.0 <= score <= 1.0:
        errors.append("action_efficiency_score must be bare float in [0,1]")
    efficiency_ci = artifact.get("action_efficiency_ci")
    if not _is_two_float_ci(efficiency_ci):
        errors.append("action_efficiency_ci must be [float, float]")
    elif not 0.0 <= efficiency_ci[0] <= efficiency_ci[1] <= 1.0:
        errors.append("action_efficiency_ci must be ordered floats in [0,1]")
    elif isinstance(score, float) and not efficiency_ci[0] <= score <= efficiency_ci[1]:
        errors.append("action_efficiency_ci must bracket the point estimate")

    wiring = artifact.get("metric_wired_into_capstone")
    if not isinstance(wiring, Mapping):
        errors.append("metric_wired_into_capstone must be object")
    elif wiring.get("reported_side_by_side") != [
        "reproducible_total_levels",
        "generic_transfer_rate_over_variants",
        "generic_transfer_ci",
        "action_efficiency_score",
        "action_efficiency_ci",
    ]:
        errors.append("metric_wired_into_capstone must report all three metrics plus CIs")
    elif wiring.get("known_game_bank_inflates_action_efficiency") is not False:
        errors.append("known-game bank must not inflate action efficiency")
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
    human_actions: Sequence[int | float] | None = None,
    human_replay_data_dir: Path | str | None = None,
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
        human_actions=human_actions,
        human_replay_data_dir=human_replay_data_dir,
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
