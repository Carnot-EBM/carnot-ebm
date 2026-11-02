"""Experiment 4556: live verifier-router generic transfer.

Spec refs: REQ-CAPSTONE-4556, SCENARIO-CAPSTONE-4556,
SCENARIO-CAPSTONE-4556-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))

from carnot import experiment_4550_honest_sprint_metric as exp4550
from carnot.agentic.arc_discriminative_router import (
    DEFAULT_CHECKPOINT_RELATIVE_PATH,
    RandomCandidateRouter,
    checkpoint_sha256,
    dominant_feature_family_from_checkpoint,
    load_cross_game_discriminative_router,
)


JsonDict = dict[str, Any]
VariantRunner = Callable[[str, Mapping[str, Any], int], Mapping[str, Any]]
VariantRunnerFactory = Callable[[str], VariantRunner]

RESULT_RELATIVE_PATH = "results/experiment_4556_verifier_router_generic_transfer.json"
EXPERIMENT_ID = "experiment_4556_verifier_router_generic_transfer"
SCHEMA = "carnot.exp4556.verifier_router_generic_transfer.v1"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4556
BASELINE_REFERENCE_RATE = 0.04
DEFAULT_VARIANT_IDS = (1,)
DEFAULT_BUDGET = 200
DEFAULT_BOOTSTRAPS = 1000
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "verifier_is_oracle",
    "generic_transfer_rate_with_verifier",
    "generic_transfer_rate_baseline",
    "generic_transfer_delta",
    "generic_transfer_ci",
    "first_contact_median_actions_with_verifier",
    "solve_rate_preserved",
    "random_router_control_passed",
    "false_negative_risk_checked",
    "null_delta_methodology_note",
    "chosen_submitted_config",
    "missing_verifier_gaps",
    "offline_reproduced",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: verifier_router_generic_transfer_<n>_above_0.04 OR "
            "complete: verifier_router_no_value_added_honest_null_gap_sharpened."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- the discriminative verifier scores cached "
            "candidate features over offline variant envs, no LLM load (1s floor)."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the DiscriminativeVerifier ranks by a LEARNED cross-game signal, "
            "oracle-DISTINCT from the executable win-check; a circular win does not count."
        )
    },
    "generic_transfer_rate_with_verifier": {
        "principle": (
            "the HEADLINE -- held-out variant transfer rate WITH the verifier-router; > 0.04 with "
            "CI-excl-baseline is the only non-circular live-value evidence."
        )
    },
    "generic_transfer_rate_baseline": {
        "principle": "0.04 -- the .420 B1 bare-solver baseline, measured the SAME way."
    },
    "generic_transfer_delta": {
        "principle": (
            "with_verifier - baseline, emitted explicitly so a null (delta 0.0) is annotated, "
            "not a control==best TAUTOLOGY false-positive."
        )
    },
    "generic_transfer_ci": {
        "principle": (
            "bootstrap CI on the transfer-rate delta; a claim above baseline requires the CI to "
            "exclude zero."
        )
    },
    "first_contact_median_actions_with_verifier": {
        "principle": (
            "the second leaderboard-relevant signal -- first-contact action efficiency on held-out "
            "variants with the router."
        )
    },
    "solve_rate_preserved": {
        "principle": "HARD gate -- the routing win must NOT drop solve-rate."
    },
    "random_router_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- the verifier-router must beat a random-router on the SAME variants."
        )
    },
    "false_negative_risk_checked": {
        "principle": "a no-value null is valid only if the random-router positive control passed."
    },
    "null_delta_methodology_note": {
        "principle": (
            "present when generic_transfer_delta==0.0 -- states the equality is an honest no-value "
            "null, not a measurement bug."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "what is recommended for SUBMITTED_AGENT_CONFIG: enable the verifier-router on a win, "
            "unchanged on null."
        )
    },
    "missing_verifier_gaps": {
        "principle": (
            "if no value-add, the sharpened discriminator still missing -- the verifier-build backlog input."
        )
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent corpus/verifier drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, DiscriminativeVerifier loadable); "
            "pre-empts missing-resource fabrication."
        )
    },
}


def _rate(solved: int, attempted: int) -> float:
    return 0.0 if attempted <= 0 else round(float(solved) / float(attempted), 10)


def _attempt_solved(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and attempt.get("solved") is True


def _first_precondition_miss(preconditions: Mapping[str, Any]) -> str | None:
    if preconditions.get("offline_arcade") is not True:
        return "offline_arcade"
    if preconditions.get("arc_value_learner_discriminative_import") is not True:
        return "discriminative_verifier_import"
    if preconditions.get("trained_verifier_loadable") is not True:
        return "trained_verifier"
    if preconditions.get("cross_game_corpus_loadable") is not True:
        return "cross_game_corpus"
    if preconditions.get("leaderboard_submission") is True:
        return "leaderboard_submission"
    return None


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - live boundary
    root_path = Path(root)
    checkpoint = root_path / DEFAULT_CHECKPOINT_RELATIVE_PATH
    corpus_summary = root_path / "results" / "arc_discriminative_verifier_v3.json"
    exp4545 = root_path / "results" / "experiment_4545_cross_game_discrimination_v3.json"
    checks: JsonDict = {
        "offline_arcade": False,
        "arc_value_learner_discriminative_import": False,
        "trained_verifier_checkpoint": str(DEFAULT_CHECKPOINT_RELATIVE_PATH),
        "trained_verifier_loadable": False,
        "trained_verifier_sha256": checkpoint_sha256(checkpoint),
        "cross_game_corpus_summary": "results/arc_discriminative_verifier_v3.json",
        "cross_game_corpus_loadable": False,
        "exp4545_artifact_present": exp4545.exists(),
        "offline_env_public_games": exp4550._public_games(root_path),
        "leaderboard_submission": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["offline_arcade_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from carnot.agentic.arc_value_learner import DiscriminativeVerifier  # noqa: F401

        checks["arc_value_learner_discriminative_import"] = True
    except Exception as exc:
        checks["arc_value_learner_discriminative_import_error"] = f"{type(exc).__name__}: {exc}"
    router = load_cross_game_discriminative_router(root=root_path)
    checks["trained_verifier_loadable"] = router is not None
    try:
        summary = json.loads(corpus_summary.read_text(encoding="utf-8"))
        checks["cross_game_corpus_loadable"] = bool(
            summary.get("feature_names") == "cross_game_features_v3"
            and int(summary.get("n_pos") or 0) > 0
            and int(summary.get("n_neg") or 0) > 0
        )
        checks["cross_game_candidate_rows"] = int(summary.get("n_pos") or 0) + int(
            summary.get("n_neg") or 0
        )
    except Exception as exc:
        checks["cross_game_corpus_error"] = f"{type(exc).__name__}: {exc}"
    checks["ok"] = _first_precondition_miss(checks) is None
    return checks


def _mode_router(mode: str, *, root: Path) -> Any | None:
    if mode == "verifier":
        return load_cross_game_discriminative_router(root=root)
    if mode == "random":
        return RandomCandidateRouter(seed=RANDOM_SEED)
    return None


def make_variant_runner(  # pragma: no cover - ARC runtime boundary
    mode: str, *, root: Path | str = REPO_ROOT
) -> VariantRunner:
    """Run the 4550 variant attempt with an optional live candidate router."""

    root_path = Path(root)
    router = _mode_router(mode, root=root_path)

    def run(game: str, spec: Mapping[str, Any], budget: int) -> JsonDict:  # pragma: no cover - ARC boundary
        from arcengine import GameAction
        from carnot import experiment_4472_variant_generic_transfer_benchmark_v4 as v4
        from carnot.agentic import arc_solver_kit as kit
        from carnot.agentic.arc_competition_agent import CarnotAgentPolicy, _level_of
        from carnot.agentic.arc_variant_generator import VariantEnv

        arc = kit.offline_arcade()
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
        policy = CarnotAgentPolicy(game, {}, force_explore=True, candidate_router=router)
        frames: list[Any] = []
        latest = None
        labels: list[str] = []
        actions = 0
        start_level: int | None = None
        reached = 0
        for _index in range(int(budget)):
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind == "RESET":
                latest = env.reset()
                if labels:
                    labels.append("RESET")
            elif kind is None:
                break
            else:
                real_data = v4._remap_reflected_data(data, spec.get("reflect"))
                latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
                labels.append(v4._action_label(int(kind), real_data))
                actions += 1
            if start_level is None:
                start_level = _level_of(latest)
            frames.append(latest)
            reached = _level_of(latest)
            if start_level is not None and reached > start_level:
                break
            if latest is None:
                break
        claimed = reached if start_level is not None and reached > start_level else 0
        gate: JsonDict = {
            "game": game,
            "reached_level": 0,
            "claimed_level": claimed,
            "reproduced": False,
            "mode": "offline_reproduction_gate_no_solution",
        }
        if claimed > 0 and labels:
            gate = dict(kit.reproduce(game, labels, v4._apply_action_label, claimed_level=claimed))
        solved = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1
        return {
            "game": game,
            "variant_signature": spec["variant_signature"],
            "variant": int(spec["variant"]),
            "kind": spec["kind"],
            "reflect": spec.get("reflect"),
            "attempted": True,
            "solved": solved,
            "reached_level": int(gate.get("reached_level") or reached) if solved else reached,
            "actions": actions,
            "solution_labels": labels if solved else [],
            "reproduction_gate": gate,
            "blocked_reason": "",
            "router_mode": mode,
        }

    return run


def default_variant_runner_factory(mode: str) -> VariantRunner:  # pragma: no cover - live boundary
    return make_variant_runner(mode, root=REPO_ROOT)


def _solved_by_signature(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(attempt.get("variant_signature")): attempt
        for attempt in attempts
        if attempt.get("attempted") is True
    }


def _median_actions(attempts: Sequence[Mapping[str, Any]]) -> float | None:
    values = [int(attempt.get("actions") or 0) for attempt in attempts if _attempt_solved(attempt)]
    if not values:
        return None
    return float(statistics.median(values))


def _paired_bootstrap_delta_ci(
    baseline_attempts: Sequence[Mapping[str, Any]],
    verifier_attempts: Sequence[Mapping[str, Any]],
    *,
    random_seed: int,
    n_bootstrap: int,
) -> list[float]:
    baseline = _solved_by_signature(baseline_attempts)
    verifier = _solved_by_signature(verifier_attempts)
    keys = sorted(set(baseline) & set(verifier))
    if not keys:
        return [0.0, 0.0]
    deltas = [
        (1.0 if _attempt_solved(verifier[key]) else 0.0)
        - (1.0 if _attempt_solved(baseline[key]) else 0.0)
        for key in keys
    ]
    if n_bootstrap <= 0:
        mean = sum(deltas) / len(deltas)
        return [round(mean, 10), round(mean, 10)]
    rng = random.Random(random_seed)
    samples = []
    n = len(deltas)
    for _ in range(int(n_bootstrap)):
        total = 0.0
        for _j in range(n):
            total += deltas[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 10), round(float(hi), 10)]


def _checksum(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _artifact_checksum_payload(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "generic_transfer_rate_with_verifier": artifact.get("generic_transfer_rate_with_verifier"),
        "generic_transfer_rate_baseline": artifact.get("generic_transfer_rate_baseline"),
        "generic_transfer_delta": artifact.get("generic_transfer_delta"),
        "generic_transfer_ci": artifact.get("generic_transfer_ci"),
        "random_router_transfer_rate": artifact.get("random_router_transfer_rate"),
        "variant_plan": artifact.get("variant_plan"),
        "newly_solved_variants": artifact.get("newly_solved_variants"),
        "preconditions_checked": artifact.get("preconditions_checked"),
    }


def _measurement(
    *,
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    runner: VariantRunner,
) -> JsonDict:
    measured = exp4550.measure_generic_transfer_over_variants(
        public_games=public_games,
        variant_ids=variant_ids,
        budget=budget,
        variant_runner=runner,
    )
    measured["first_contact_median_actions"] = _median_actions(measured["variant_attempts"])
    return measured


def _blocked_artifact(
    *,
    resource: str,
    preconditions: Mapping[str, Any],
    public_games: Sequence[str],
    variant_ids: Sequence[int],
    budget: int,
    dominant_weight_family: str,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4556",
            "SCENARIO-CAPSTONE-4556",
            "SCENARIO-CAPSTONE-4556-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"complete: blocked_{resource}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_verifier": 0.0,
        "generic_transfer_rate_baseline": BASELINE_REFERENCE_RATE,
        "generic_transfer_delta": 0.0,
        "generic_transfer_ci": [0.0, 0.0],
        "first_contact_median_actions_with_verifier": None,
        "solve_rate_preserved": False,
        "random_router_transfer_rate": 0.0,
        "random_router_control_passed": False,
        "false_negative_risk_checked": False,
        "null_delta_methodology_note": (
            "blocked before measurement; no verifier-router delta was fabricated."
        ),
        "chosen_submitted_config": "unchanged",
        "missing_verifier_gaps": [f"blocked_{resource}; strongest_weight_family={dominant_weight_family}"],
        "offline_reproduced": False,
        "newly_solved_variants": [],
        "preconditions_checked": dict(preconditions),
        "variant_plan": {
            "public_games": sorted(str(game) for game in public_games),
            "public_game_count": len(public_games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
        },
        "baseline_measurement": {},
        "verifier_measurement": {},
        "random_router_measurement": {},
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum(_artifact_checksum_payload(artifact))
    return artifact


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    public_games: Sequence[str] | None = None,
    variant_ids: Sequence[int] = DEFAULT_VARIANT_IDS,
    budget: int = DEFAULT_BUDGET,
    preconditions_checked: Mapping[str, Any] | None = None,
    variant_runner_factory: VariantRunnerFactory = default_variant_runner_factory,
    n_bootstrap: int = DEFAULT_BOOTSTRAPS,
    dominant_weight_family: str | None = None,
) -> JsonDict:
    started = time.monotonic()
    root_path = Path(root)
    preconditions = dict(preconditions_checked or check_preconditions(root_path))
    games = list(public_games or preconditions.get("offline_env_public_games") or [])
    checkpoint = root_path / DEFAULT_CHECKPOINT_RELATIVE_PATH
    if dominant_weight_family is None:
        try:
            dominant_weight_family = dominant_feature_family_from_checkpoint(checkpoint)
        except Exception:
            dominant_weight_family = "unknown"

    miss = _first_precondition_miss(preconditions)
    if miss:
        return _blocked_artifact(
            resource=miss,
            preconditions=preconditions,
            public_games=games,
            variant_ids=variant_ids,
            budget=budget,
            dominant_weight_family=dominant_weight_family,
        )

    baseline = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("baseline"),
    )
    verifier = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("verifier"),
    )
    random_control = _measurement(
        public_games=games,
        variant_ids=variant_ids,
        budget=budget,
        runner=variant_runner_factory("random"),
    )

    baseline_rate = float(baseline["generic_transfer_rate_over_variants"])
    verifier_rate = float(verifier["generic_transfer_rate_over_variants"])
    random_rate = float(random_control["generic_transfer_rate_over_variants"])
    delta = round(verifier_rate - baseline_rate, 10)
    delta_ci = _paired_bootstrap_delta_ci(
        baseline["variant_attempts"],
        verifier["variant_attempts"],
        random_seed=RANDOM_SEED,
        n_bootstrap=n_bootstrap,
    )
    solve_rate_preserved = verifier_rate >= baseline_rate
    random_router_control_passed = verifier_rate > random_rate

    baseline_by_sig = _solved_by_signature(baseline["variant_attempts"])
    newly_solved = []
    newly_reproduced = []
    for attempt in verifier["variant_attempts"]:
        signature = str(attempt.get("variant_signature"))
        if _attempt_solved(attempt) and not _attempt_solved(baseline_by_sig.get(signature, {})):
            newly_solved.append(signature)
            gate = attempt.get("reproduction_gate")
            newly_reproduced.append(isinstance(gate, Mapping) and gate.get("reproduced") is True)
    offline_reproduced = all(newly_reproduced) if newly_reproduced else True
    wins = (
        verifier_rate > BASELINE_REFERENCE_RATE
        and solve_rate_preserved
        and random_router_control_passed
        and offline_reproduced
        and delta_ci[0] > 0.0
    )
    null_note = (
        "generic_transfer_delta==0.0 is an honest no-value null under the paired same-variant "
        "measurement, not a control==best tautology."
        if delta == 0.0
        else ""
    )
    missing_gaps = (
        []
        if wins
        else [f"verifier_router_no_value_added; strongest_weight_family={dominant_weight_family}"]
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": [
            "REQ-CAPSTONE-4556",
            "SCENARIO-CAPSTONE-4556",
            "SCENARIO-CAPSTONE-4556-FIELD-PRINCIPLES",
        ],
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": (
            f"success: verifier_router_generic_transfer_{verifier_rate:.3f}_above_0.04"
            if wins
            else "complete: verifier_router_no_value_added_honest_null_gap_sharpened"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_verifier": verifier_rate,
        "generic_transfer_rate_baseline": baseline_rate,
        "generic_transfer_baseline_reference": BASELINE_REFERENCE_RATE,
        "generic_transfer_delta": delta,
        "generic_transfer_ci": delta_ci,
        "first_contact_median_actions_with_verifier": verifier["first_contact_median_actions"],
        "first_contact_median_actions_baseline": baseline["first_contact_median_actions"],
        "random_router_transfer_rate": random_rate,
        "first_contact_median_actions_random_router": random_control["first_contact_median_actions"],
        "solve_rate_preserved": solve_rate_preserved,
        "random_router_control_passed": random_router_control_passed,
        "false_negative_risk_checked": random_router_control_passed,
        "null_delta_methodology_note": null_note,
        "chosen_submitted_config": "enable_verifier_router" if wins else "unchanged",
        "missing_verifier_gaps": missing_gaps,
        "dominant_verifier_weight_family": dominant_weight_family,
        "offline_reproduced": offline_reproduced,
        "newly_solved_variants": sorted(newly_solved),
        "preconditions_checked": preconditions,
        "variant_plan": {
            "public_games": sorted(str(game) for game in games),
            "public_game_count": len(games),
            "variant_ids": [int(item) for item in variant_ids],
            "variants_per_game": len(variant_ids),
            "budget": int(budget),
            "runner": "generic_solver_offline_variant_env",
            "arms": ["baseline", "verifier", "random"],
        },
        "baseline_measurement": baseline,
        "verifier_measurement": verifier,
        "random_router_measurement": random_control,
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.monotonic() - started, 6),
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
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in (
        "generic_transfer_rate_with_verifier",
        "generic_transfer_rate_baseline",
        "generic_transfer_delta",
    ):
        if not isinstance(artifact.get(field), float):
            errors.append(f"{field} must be a bare float")
    for field in ("solve_rate_preserved", "random_router_control_passed", "offline_reproduced"):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    ci = artifact.get("generic_transfer_ci")
    if (
        not isinstance(ci, list)
        or len(ci) != 2
        or not all(isinstance(value, float) for value in ci)
    ):
        errors.append("generic_transfer_ci must be [float, float]")
    if artifact.get("generic_transfer_delta") == 0.0 and not artifact.get(
        "null_delta_methodology_note"
    ):
        errors.append("null_delta_methodology_note required for zero delta")
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
    write: bool = True,
) -> JsonDict:
    artifact = build_artifact(root)
    if write:
        write_artifact(root, artifact=artifact)
    return artifact


def main() -> int:  # pragma: no cover - requested command boundary
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - requested command boundary
    raise SystemExit(main())
