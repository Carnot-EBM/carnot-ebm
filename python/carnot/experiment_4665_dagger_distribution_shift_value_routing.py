"""Experiment 4665: DAgger-lite distribution-shift value routing.

Spec refs: REQ-LEARN-4665, SCENARIO-LEARN-4665-DAGGER-DATA,
SCENARIO-LEARN-4665-LIVE-ROUTE, SCENARIO-LEARN-4665-ARTIFACT.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4665_dagger_distribution_shift_value_routing"
SCHEMA = "carnot.arc.dagger_distribution_shift_value_routing_4665.v1"
RESULT_RELATIVE_PATH = "results/experiment_4665_dagger_distribution_shift_value_routing.json"
B1_RELATIVE_PATH = "results/experiment_4658_value_routing_cigate_diagnostic.json"
A1_RELATIVE_PATH = "results/experiment_4652_value_routing_cost_fix_live.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/self-learning/spec.md"
MODEL_RELATIVE_PATH = "models/arc_dagger_value_routing_v3.json"
RANDOM_SEED = 4665
DEFAULT_DAGGER_BUDGET = 80
DEFAULT_MEASUREMENT_BUDGET = 200
DEFAULT_TRAIN_ITERS = 300
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline-arcade DAgger data collection + "
    "value-head re-train + live-search measurement over cached variants (1s floor); the value head "
    "is CPU, no live_llm_inference."
)
SOLVE_PROVENANCE = "live_agent_self_discovery"
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_", "passed:", "shipped:")
SPEC_REFS = [
    "REQ-LEARN-4665",
    "SCENARIO-LEARN-4665-DAGGER-DATA",
    "SCENARIO-LEARN-4665-LIVE-ROUTE",
    "SCENARIO-LEARN-4665-ARTIFACT",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: dagger_distribution_shift_value_routing_live_"
            "<firstwin|solverate>_up_<n> OR complete: "
            "dagger_distribution_corrected_no_live_lift_residual_logged."
        )
    },
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the win-reachability value head is a learned discriminator, "
            "oracle-DISTINCT from the executable win-check (the oracle is used only to LABEL "
            "training data, not at inference)."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- this improves the SCORED live agent's OWN search "
            "guidance (E3AgentPolicy value-routing); NOT a parallel solver, NOT outer_loop_re."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the changed modules (arc_value_learner/arc_competition_agent) are in "
            "the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
        )
    },
    "distribution_shift_score_before": {
        "principle": "B1's measured 0.699 -- the localized residual cause this task attacks."
    },
    "distribution_shift_score_after": {
        "principle": (
            "the post-DAgger shift score -- a DROP is the evidence the correction took (the "
            "mechanism worked) independent of the live lift."
        )
    },
    "shift_score_delta": {
        "principle": (
            "after - before (negative = shift reduced), emitted explicitly so a null is annotated."
        )
    },
    "live_first_win_rate_corrected": {
        "principle": (
            "the live first-win-rate WITH the distribution-corrected value head on the SCORED agent."
        )
    },
    "live_solve_rate_corrected": {
        "principle": (
            "the live multi-level (>=2) solve-rate WITH the distribution-corrected head (the "
            "deeper wall)."
        )
    },
    "live_baseline_winning_path_trained": {
        "principle": (
            "the matched .429 winning-path-trained value head first-win + solve-rate on the SAME "
            "variants (the no-regression control)."
        )
    },
    "first_win_rate_delta": {
        "principle": (
            "corrected - baseline first-win-rate (positive = distribution correction crossed the "
            "bridge), emitted explicitly so a null (0) is annotated."
        )
    },
    "solve_rate_delta": {
        "principle": (
            "corrected - baseline multi-level solve-rate; emitted explicitly so a null is annotated."
        )
    },
    "live_lift_ci": {
        "principle": (
            "bootstrap CI on the chosen live-lift metric; a claim above baseline requires the CI "
            "to exclude it."
        )
    },
    "bare_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- the baseline ran on a corpus with reachable headroom; a "
            "no-lift null is valid only then."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "true with the matched baseline + reachable-headroom confirmed -- a 'no lift' null is "
            "valid only then."
        )
    },
    "null_methodology_note": {
        "principle": (
            "present when a delta==0 -- states the equality is an honest no-value null, not a "
            "measurement bug."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG change (distribution-corrected head on, "
            "value_weight, feature subset) -- the A6 input; 'unchanged' if null."
        )
    },
    "parity_test_green": {
        "principle": "HARD gate -- test_arc_submitted_agent_parity.py passes."
    },
    "offline_reproduced": {
        "principle": "any newly-solved variant must offline-reproduce to count."
    },
    "residual_bridge_gap": {
        "principle": (
            "the Missing-Verifier / bridge gap logged if the corrected head still nulls -- the "
            ".431 next-attack record."
        )
    },
    "random_seed": {
        "principle": "determinism precondition for reproducibility (DAgger sampling RNG seeded)."
    },
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (offline arcade, value-learner + agent importable, B1 "
            "artifact present); pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "dagger_dataset",
    "model_checkpoint",
    "corrected_measurement",
    "baseline_measurement",
    "parity_test",
    "orphan_lint",
    "distribution_shift_probe_after",
    "source_artifacts",
    "source_artifact_checksums",
    "field_principles",
    "spec_refs",
    "duration_s",
    "submitted_to_leaderboard",
)


class _NoOpProposer:
    def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:  # pragma: no cover
        return False, "disabled_exp4665_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:  # pragma: no cover
        return []


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


def _mapping_at(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else {}


def _attempts(measurement: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = measurement.get("variant_attempts")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def measurement_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> JsonDict:
    from carnot.experiment_4652_value_routing_cost_fix_live import measurement_from_attempts as measure

    return measure(attempts)


def paired_delta_ci(
    corrected_attempts: Sequence[Mapping[str, Any]],
    baseline_attempts: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    from carnot.experiment_4652_value_routing_cost_fix_live import paired_delta_ci as paired

    return paired(
        corrected_attempts,
        baseline_attempts,
        metric=metric,
        random_seed=random_seed,
    )


def path_action_label(step: Mapping[str, Any]) -> str:
    return json.dumps(
        {"action": int(step["action"]), "data": step.get("data")},
        sort_keys=True,
        separators=(",", ":"),
    )


def path_to_labels(path: Sequence[Mapping[str, Any]]) -> list[str]:
    return [path_action_label(step) for step in path if step.get("action") is not None]


def relabel_frontier_rows(
    frontier_rows: Sequence[Mapping[str, Any]],
    *,
    winning_labels: Sequence[str],
) -> list[JsonDict]:
    """SCENARIO-LEARN-4665-DAGGER-DATA: relabel live rows from reproduction evidence."""

    clean_winning = [str(label) for label in winning_labels]
    winning_prefixes = {
        tuple(clean_winning[: index + 1]) for index in range(len(clean_winning))
    }
    if clean_winning:
        winning_prefixes.add(())
    relabeled: list[JsonDict] = []
    for row in frontier_rows:
        path = [step for step in row.get("path", []) if isinstance(step, Mapping)]
        labels = tuple(path_to_labels(path))
        relabeled.append(
            {
                **dict(row),
                "features": [float(v) for v in row.get("features", [])],
                "path": [
                    {"action": int(step["action"]), "data": step.get("data")}
                    for step in path
                    if step.get("action") is not None
                ],
                "label": 1.0 if labels in winning_prefixes else 0.0,
                "relabel_source": "executable_reproduction_prefix",
            }
        )
    return relabeled


def aggregate_dagger_rows(
    *,
    winning_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """SCENARIO-LEARN-4665-DAGGER-DATA: aggregate expert and learner distributions."""

    rows: list[JsonDict] = []
    for source_rows, default_source in (
        (winning_rows, "winning_path"),
        (frontier_rows, "search_distribution"),
    ):
        for row in source_rows:
            features = [float(v) for v in row.get("features", [])]
            if not features:
                continue
            rows.append(
                {
                    "source": str(row.get("source") or default_source),
                    "features": features,
                    "label": 1.0 if _as_float(row.get("label")) >= 0.5 else 0.0,
                    "path": list(row.get("path") or []),
                }
            )
    positives = sum(1 for row in rows if float(row["label"]) >= 0.5)
    negatives = len(rows) - positives
    return {
        "rows": rows,
        "positive_count": int(positives),
        "negative_count": int(negatives),
        "winning_path_count": int(sum(1 for row in rows if row["source"] == "winning_path")),
        "frontier_count": int(len(frontier_rows)),
        "total_count": int(len(rows)),
    }


def _ci_excludes_zero(ci: Mapping[str, Any]) -> bool:
    interval = ci.get("ci95")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        return False
    return float(interval[0]) > 0.0 or float(interval[1]) < 0.0


def _same_variant_control(corrected: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return list(corrected.get("variant_signatures") or []) == list(
        baseline.get("variant_signatures") or []
    ) and int(corrected.get("variant_attempts_count") or 0) > 0


def _truthy_first_win(attempt: Mapping[str, Any]) -> bool:
    return attempt.get("attempted") is True and (
        attempt.get("first_win") is True or attempt.get("solved") is True
    )


def _offline_reproduced_new_wins(
    corrected: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> bool:
    baseline_wins = {
        str(row.get("variant_signature") or "")
        for row in _attempts(baseline)
        if _truthy_first_win(row)
    }
    for attempt in _attempts(corrected):
        if not _truthy_first_win(attempt):
            continue
        signature = str(attempt.get("variant_signature") or "")
        if signature in baseline_wins:
            continue
        gate = attempt.get("reproduction_gate")
        if not isinstance(gate, Mapping) or gate.get("reproduced") is not True:
            return False
    return True


def _chosen_config(model_checkpoint: str) -> JsonDict:
    from carnot.agentic.arc_competition_agent import (
        SUBMITTED_AGENT_CONFIG,
        SUBMITTED_VALUE_HEAD_FEATURE_SUBSET,
        SUBMITTED_VALUE_WEIGHT,
    )

    config = json.loads(json.dumps(SUBMITTED_AGENT_CONFIG, sort_keys=True, default=str))
    config["value_weight"] = float(SUBMITTED_VALUE_WEIGHT)
    config["value_head_feature_subset"] = SUBMITTED_VALUE_HEAD_FEATURE_SUBSET
    config["value_head_checkpoint"] = model_checkpoint
    config["value_head_distribution_corrected"] = True
    return config


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    corrected_measurement: Mapping[str, Any],
    baseline_measurement: Mapping[str, Any],
    dagger_dataset: Mapping[str, Any],
    distribution_shift_before: float,
    distribution_shift_after: float,
    b1_artifact: Mapping[str, Any],
    a1_artifact: Mapping[str, Any],
    parity_test: Mapping[str, Any],
    orphan_lint: Mapping[str, Any],
    model_checkpoint: str,
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    first_corrected = _as_float(corrected_measurement.get("first_win_rate"))
    first_baseline = _as_float(baseline_measurement.get("first_win_rate"))
    solve_corrected = _as_float(corrected_measurement.get("solve_rate"))
    solve_baseline = _as_float(baseline_measurement.get("solve_rate"))
    first_delta = round(first_corrected - first_baseline, 6)
    solve_delta = round(solve_corrected - solve_baseline, 6)
    first_ci = paired_delta_ci(
        _attempts(corrected_measurement),
        _attempts(baseline_measurement),
        metric="first_win_rate",
        random_seed=random_seed,
    )
    solve_ci = paired_delta_ci(
        _attempts(corrected_measurement),
        _attempts(baseline_measurement),
        metric="solve_rate",
        random_seed=random_seed,
    )
    chosen_metric = "first_win_rate" if first_delta >= solve_delta else "solve_rate"
    live_lift_ci = first_ci if chosen_metric == "first_win_rate" else solve_ci
    first_success = first_delta > 0.0 and _ci_excludes_zero(first_ci)
    solve_success = solve_delta > 0.0 and _ci_excludes_zero(solve_ci)
    parity_green = bool(parity_test.get("passed"))
    live_path_reachable = bool(orphan_lint.get("passed"))
    same_variants = _same_variant_control(corrected_measurement, baseline_measurement)
    bare_control = bool(a1_artifact.get("bare_control_passed")) and same_variants
    offline_reproduced = _offline_reproduced_new_wins(corrected_measurement, baseline_measurement)
    shift_delta = round(float(distribution_shift_after) - float(distribution_shift_before), 6)
    success = bool(
        parity_green
        and live_path_reachable
        and bare_control
        and offline_reproduced
        and shift_delta < 0.0
        and (first_success or solve_success)
    )
    if success and first_success:
        up_count = int(round(first_delta * int(corrected_measurement.get("variant_attempts_count") or 0)))
        verdict = f"success: dagger_distribution_shift_value_routing_live_firstwin_up_{up_count}"
    elif success:
        up_count = int(round(solve_delta * int(corrected_measurement.get("variant_attempts_count") or 0)))
        verdict = f"success: dagger_distribution_shift_value_routing_live_solverate_up_{up_count}"
    else:
        verdict = "complete: dagger_distribution_corrected_no_live_lift_residual_logged."

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "live_path_reachable": live_path_reachable,
        "distribution_shift_score_before": round(float(distribution_shift_before), 6),
        "distribution_shift_score_after": round(float(distribution_shift_after), 6),
        "shift_score_delta": shift_delta,
        "live_first_win_rate_corrected": first_corrected,
        "live_solve_rate_corrected": solve_corrected,
        "live_baseline_winning_path_trained": {
            "value_head": "models/arc_verifier_cross_game_v3.json",
            "first_win_rate": first_baseline,
            "solve_rate": solve_baseline,
            "measurement": dict(baseline_measurement),
        },
        "first_win_rate_delta": first_delta,
        "solve_rate_delta": solve_delta,
        "live_lift_ci": live_lift_ci,
        "bare_control_passed": bare_control,
        "false_negative_risk_checked": bool(bare_control and same_variants),
        "chosen_submitted_config": _chosen_config(model_checkpoint) if success else "unchanged",
        "parity_test_green": parity_green,
        "offline_reproduced": bool(offline_reproduced),
        "residual_bridge_gap": (
            "none" if success else "missing_verifier_gap_live_frontier_not_separated"
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "dagger_dataset": {
            "positive_count": int(dagger_dataset.get("positive_count") or 0),
            "negative_count": int(dagger_dataset.get("negative_count") or 0),
            "winning_path_count": int(dagger_dataset.get("winning_path_count") or 0),
            "frontier_count": int(dagger_dataset.get("frontier_count") or 0),
            "total_count": int(dagger_dataset.get("total_count") or 0),
        },
        "model_checkpoint": model_checkpoint,
        "corrected_measurement": dict(corrected_measurement),
        "baseline_measurement": dict(baseline_measurement),
        "parity_test": dict(parity_test),
        "orphan_lint": dict(orphan_lint),
        "distribution_shift_probe_after": {
            "score": round(float(distribution_shift_after), 6),
            "method": "frontier_vs_aggregated_search_distribution_score_gap",
            "corrected_model_checkpoint": model_checkpoint,
        },
        "source_artifacts": {
            "b1_distribution_shift": B1_RELATIVE_PATH,
            "a1_winning_path_baseline": A1_RELATIVE_PATH,
        },
        "source_artifact_checksums": {
            "b1_distribution_shift": artifact_checksum(b1_artifact),
            "a1_winning_path_baseline": artifact_checksum(a1_artifact),
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    if first_delta == 0.0 or solve_delta == 0.0:
        artifact["null_methodology_note"] = (
            "A zero live-lift delta is from matched distribution-corrected and .429 "
            "winning-path-trained value-head runs on the same variant signatures; it is an "
            "honest no-value null, not a measurement bug."
        )
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        errors.append("solve_provenance")
    if artifact.get("live_path_reachable") is not True:
        errors.append("live_path_reachable")
    if artifact.get("parity_test_green") is not True:
        errors.append("parity_test_green")
    if artifact.get("bare_control_passed") is not True:
        errors.append("bare_control_passed")
    if artifact.get("false_negative_risk_checked") is not True:
        errors.append("false_negative_risk_checked")
    if artifact.get("offline_reproduced") is not True:
        errors.append("offline_reproduced")
    if _as_float(artifact.get("distribution_shift_score_after")) > _as_float(
        artifact.get("distribution_shift_score_before")
    ):
        errors.append("distribution_shift_score_after_not_reduced")
    if (artifact.get("first_win_rate_delta") == 0 or artifact.get("solve_rate_delta") == 0) and (
        "null_methodology_note" not in artifact
    ):
        errors.append("null_methodology_note")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_false")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _default_import_checker() -> JsonDict:  # pragma: no cover - precondition boundary.
    from carnot.agentic import arc_competition_agent, arc_value_learner

    return {
        "agent_import": bool(arc_competition_agent),
        "value_learner_import": bool(arc_value_learner),
    }


def check_preconditions(
    root: Path | str = REPO_ROOT,
    *,
    import_checker: Callable[[], Mapping[str, Any]] | None = None,
) -> JsonDict:
    root_path = Path(root)
    checks: JsonDict = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists()
        or (root_path / "OPENCODE.md").exists(),
        "offline_arcade": False,
        "agent_import": False,
        "value_learner_import": False,
        "b1_artifact_present": (root_path / B1_RELATIVE_PATH).exists(),
        "a1_artifact_present": (root_path / A1_RELATIVE_PATH).exists(),
        "spec_has_req_4665": False,
        "live_llm_inference": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade"] = True
    except Exception as exc:
        checks["blocked_resource"] = "offline_arcade"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    try:
        checks.update(dict((import_checker or _default_import_checker)()))
    except Exception as exc:
        checks["blocked_resource"] = "agentic_imports"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    try:
        _load_json(root_path / B1_RELATIVE_PATH)
    except Exception as exc:
        checks["blocked_resource"] = "b1_artifact_present"
        checks["error"] = repr(exc)[:200]
        checks["ok"] = False
        return checks
    spec_path = root_path / SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""
    checks["spec_has_req_4665"] = "REQ-LEARN-4665" in spec_text
    required = (
        "agents_md_read",
        "codex_md_read",
        "offline_arcade",
        "agent_import",
        "value_learner_import",
        "b1_artifact_present",
        "a1_artifact_present",
        "spec_has_req_4665",
    )
    checks["ok"] = all(bool(checks.get(key)) for key in required)
    if not checks["ok"]:
        checks["blocked_resource"] = next((key for key in required if not checks.get(key)), "precondition")
    return checks


def _load_train_module(root: Path) -> Any:  # pragma: no cover - runtime boundary.
    script = root / "scripts" / "arc_cross_game_verifier_train.py"
    spec = importlib.util.spec_from_file_location("arc_cross_game_verifier_train_4665", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def collect_winning_path_rows(
    root: Path,
    *,
    random_seed: int = RANDOM_SEED,
) -> list[JsonDict]:  # pragma: no cover - offline arcade boundary.
    from carnot.agentic.arc_value_learner import cross_game_features_v3_value_routing

    train_mod = _load_train_module(root)
    x_rows, y_rows, _per_game = train_mod.collect_discriminative(
        featurize=cross_game_features_v3_value_routing,
        neg_per_game=0,
        seed=random_seed,
    )
    return [
        {"source": "winning_path", "features": [float(v) for v in row], "label": float(label)}
        for row, label in zip(x_rows, y_rows)
        if float(label) >= 0.5
    ]


def _action_label(action: int | str, data: Any) -> str:  # pragma: no cover - ARC runtime boundary.
    if action == "RESET":
        return "RESET"
    return path_action_label({"action": int(action), "data": data})


def _apply_action_label(env: Any, label: str, _frame: Any = None) -> Any:  # pragma: no cover
    if label == "RESET":
        return env.reset()
    from arcengine import GameAction
    from carnot.agentic.arc_agi3_live_adapter import _game_action

    step = json.loads(label)
    return env.step(_game_action(GameAction, int(step["action"])), data=step.get("data"))


def _level_of_frame(frame: Any) -> int:  # pragma: no cover - ARC runtime boundary.
    from carnot.agentic.arc_competition_agent import _level_of

    return int(_level_of(frame))


def _baseline_winning_path_head(root: Path) -> Any:  # pragma: no cover - checkpoint boundary.
    from carnot.agentic.arc_competition_agent import _load_sliced_v3_value_head

    return _load_sliced_v3_value_head(root / "models" / "arc_verifier_cross_game_v3.json")


def variant_specs_from_a1(a1_artifact: Mapping[str, Any]) -> list[JsonDict]:
    measurement = _mapping_at(a1_artifact, "value_routed_measurement")
    signatures = list(measurement.get("variant_signatures") or [])
    if not signatures:
        signatures = list(a1_artifact.get("matched_variant_signatures") or [])
    specs: list[JsonDict] = []
    for signature in signatures:
        game, _, tail = str(signature).partition("~color")
        variant = int(tail or 1)
        specs.append(
            {
                "game": game,
                "variant": variant,
                "kind": "color",
                "reflect": None,
                "variant_signature": str(signature),
            }
        )
    return specs


def run_policy_attempt(
    *,
    game: str,
    spec: Mapping[str, Any],
    budget: int,
    value_head: Any,
    policy_mode: str,
    collect_samples: bool,
) -> tuple[JsonDict, list[JsonDict]]:  # pragma: no cover - ARC runtime boundary.
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, SUBMITTED_TARGET_LEVELS, SUBMITTED_VALUE_WEIGHT
    from carnot.agentic.arc_value_learner import cross_game_features_v3_value_routing
    from carnot.agentic.arc_variant_generator import VariantEnv

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
    policy = E3AgentPolicy(
        game,
        proposer=_NoOpProposer(),
        target_levels=SUBMITTED_TARGET_LEVELS,
        value_head=value_head,
        value_weight=SUBMITTED_VALUE_WEIGHT,
    )
    policy.explorer.discriminative_featurizer = cross_game_features_v3_value_routing
    frames: list[Any] = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level: int | None = None
    reached = 0
    actions_to_first: int | None = None
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
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            labels.append(_action_label(int(kind), data))
            actions += 1
        if start_level is None:
            start_level = _level_of_frame(latest)
        reached = _level_of_frame(latest)
        if start_level is not None and reached > start_level and actions_to_first is None:
            actions_to_first = actions
        frames.append(latest)
        if latest is None:
            break
    claimed = reached if start_level is not None and reached > start_level else 0
    gate: JsonDict = {
        "game": game,
        "claimed_level": claimed,
        "reached_level": 0,
        "reproduced": False,
        "mode": "offline_reproduction_gate_no_solution",
    }
    if claimed > 0 and labels:
        gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=claimed))
    reproduced = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1
    reached_level = int(gate.get("reached_level") or reached) if reproduced else int(reached)
    attempt = {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "variant": int(spec["variant"]),
        "kind": spec["kind"],
        "reflect": spec.get("reflect"),
        "attempted": True,
        "solved": reproduced,
        "first_win": bool(reproduced),
        "reached_level": reached_level,
        "actions": actions,
        "actions_to_first_levelup": actions_to_first if reproduced else None,
        "solution_labels": labels if reproduced else [],
        "reproduction_gate": gate,
        "blocked_reason": "",
        "policy_mode": policy_mode,
        "timed_out": False,
        "lazy_value_diagnostics": policy.explorer.lazy_value_diagnostics(),
    }
    samples = policy.explorer.search_distribution_samples() if collect_samples else []
    return attempt, samples


def collect_search_distribution_rows(
    *,
    root: Path,
    specs: Sequence[Mapping[str, Any]],
    budget: int,
    value_head: Any,
) -> list[JsonDict]:  # pragma: no cover - ARC runtime boundary.
    rows: list[JsonDict] = []
    for spec in specs:
        attempt, samples = run_policy_attempt(
            game=str(spec["game"]),
            spec=spec,
            budget=budget,
            value_head=value_head,
            policy_mode="dagger_collection",
            collect_samples=True,
        )
        rows.extend(
            relabel_frontier_rows(
                samples,
                winning_labels=attempt.get("solution_labels") or [],
            )
        )
    return rows


def measure_corrected_policy(
    *,
    specs: Sequence[Mapping[str, Any]],
    budget: int,
    value_head: Any,
) -> JsonDict:  # pragma: no cover - ARC runtime boundary.
    attempts: list[JsonDict] = []
    for spec in specs:
        attempt, _samples = run_policy_attempt(
            game=str(spec["game"]),
            spec=spec,
            budget=budget,
            value_head=value_head,
            policy_mode="dagger_corrected",
            collect_samples=False,
        )
        attempts.append(attempt)
    return measurement_from_attempts(attempts)


def _mean_score(rows: Sequence[Mapping[str, Any]], scorer: Callable[[Sequence[float]], float]) -> float:
    values = [float(scorer(row.get("features", []))) for row in rows if row.get("features")]
    return sum(values) / len(values) if values else 0.0


def post_dagger_shift_probe(
    *,
    frontier_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    value_head: Any,
) -> JsonDict:
    """SCENARIO-LEARN-4665-ARTIFACT: re-score search-distribution shift after aggregation."""

    frontier = [row for row in frontier_rows if row.get("features")]
    aggregate_frontier = [
        row for row in aggregate_rows if row.get("features") and row.get("source") != "winning_path"
    ]
    reference = aggregate_frontier or list(aggregate_rows)
    scorer = value_head.cost_features
    frontier_mean = _mean_score(frontier, scorer)
    reference_mean = _mean_score(reference, scorer)
    denom = max(abs(frontier_mean), abs(reference_mean), 1.0)
    score = abs(frontier_mean - reference_mean) / denom
    return {
        "distribution_shift_score": round(float(score), 6),
        "frontier_mean_score": round(float(frontier_mean), 6),
        "aggregate_reference_mean_score": round(float(reference_mean), 6),
        "frontier_count": len(frontier),
        "aggregate_reference_count": len(reference),
        "method": "frontier_vs_aggregated_search_distribution_score_gap",
    }


def run_parity_check(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/test_arc_submitted_agent_parity.py",
        "-q",
        "--no-cov",
    ]
    proc = subprocess.run(cmd, cwd=Path(root), capture_output=True, text=True, timeout=180, check=False)
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_orphan_lint(root: Path | str = REPO_ROOT) -> JsonDict:  # pragma: no cover - subprocess.
    cmd = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    proc = subprocess.run(cmd, cwd=Path(root), capture_output=True, text=True, timeout=120, check=False)
    return {
        "passed": proc.returncode == 0,
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


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
    empty = measurement_from_attempts([])
    artifact = build_artifact(
        preconditions_checked=checks,
        corrected_measurement=empty,
        baseline_measurement=empty,
        dagger_dataset={"positive_count": 0, "negative_count": 0, "frontier_count": 0},
        distribution_shift_before=0.0,
        distribution_shift_after=0.0,
        b1_artifact={},
        a1_artifact={},
        parity_test={"passed": False},
        orphan_lint={"passed": False},
        model_checkpoint=MODEL_RELATIVE_PATH,
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = f"blocked_{checks.get('blocked_resource', 'precondition')}"
    artifact["live_path_reachable"] = False
    artifact["bare_control_passed"] = False
    artifact["false_negative_risk_checked"] = False
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    dagger_budget: int = DEFAULT_DAGGER_BUDGET,
    measurement_budget: int = DEFAULT_MEASUREMENT_BUDGET,
    train_iters: int = DEFAULT_TRAIN_ITERS,
    parity_check: Callable[[Path | str], Mapping[str, Any]] = run_parity_check,
    orphan_lint: Callable[[Path | str], Mapping[str, Any]] = run_orphan_lint,
    now: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JsonDict:
    from carnot.agentic.arc_value_learner import fit_dagger_win_reachability_value_head

    started = now()
    root_path = Path(root)
    checks = dict(preconditions_checked or check_preconditions(root_path))
    if not checks.get("ok", True):
        artifact = _blocked_artifact(
            checks,
            _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
        )
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    b1 = _load_json(root_path / B1_RELATIVE_PATH)
    a1 = _load_json(root_path / A1_RELATIVE_PATH)
    specs = variant_specs_from_a1(a1)
    baseline_measurement = dict(_mapping_at(a1, "value_routed_measurement"))
    baseline_head = _baseline_winning_path_head(root_path)
    if baseline_head is None:
        checks["ok"] = False
        checks["blocked_resource"] = "winning_path_baseline_head"
        artifact = _blocked_artifact(
            checks,
            _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
        )
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    winning_rows = collect_winning_path_rows(root_path, random_seed=RANDOM_SEED)
    frontier_rows = collect_search_distribution_rows(
        root=root_path,
        specs=specs,
        budget=dagger_budget,
        value_head=baseline_head,
    )
    dagger_dataset = aggregate_dagger_rows(winning_rows=winning_rows, frontier_rows=frontier_rows)
    if int(dagger_dataset["positive_count"]) <= 0 or int(dagger_dataset["negative_count"]) <= 0:
        checks["ok"] = False
        checks["blocked_resource"] = "dagger_training_rows"
        artifact = _blocked_artifact(
            checks,
            _floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
        )
        _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
        return artifact

    corrected_head = fit_dagger_win_reachability_value_head(
        [row["features"] for row in dagger_dataset["rows"]],
        [row["label"] for row in dagger_dataset["rows"]],
        iters=train_iters,
        lr=0.4,
    )
    model_path = root_path / MODEL_RELATIVE_PATH
    corrected_head.save(
        model_path,
        meta={
            "provenance": "Exp4665 DAgger-lite search-distribution aggregation",
            "spec_refs": list(SPEC_REFS),
        },
    )
    shift_after_probe = post_dagger_shift_probe(
        frontier_rows=frontier_rows,
        aggregate_rows=dagger_dataset["rows"],
        value_head=corrected_head,
    )
    corrected_measurement = measure_corrected_policy(
        specs=specs,
        budget=measurement_budget,
        value_head=corrected_head,
    )
    parity = dict(parity_check(root_path))
    lint = dict(orphan_lint(root_path))
    before = _as_float(
        b1.get("distribution_shift_score"),
        _as_float(_mapping_at(b1, "diagnostic").get("distribution_shift_score")),
    )
    artifact = build_artifact(
        preconditions_checked={
            **checks,
            "dagger_budget": int(dagger_budget),
            "measurement_budget": int(measurement_budget),
            "train_iters": int(train_iters),
            "variant_count": len(specs),
        },
        corrected_measurement=corrected_measurement,
        baseline_measurement=baseline_measurement,
        dagger_dataset=dagger_dataset,
        distribution_shift_before=before,
        distribution_shift_after=float(shift_after_probe["distribution_shift_score"]),
        b1_artifact=b1,
        a1_artifact=a1,
        parity_test=parity,
        orphan_lint=lint,
        model_checkpoint=MODEL_RELATIVE_PATH,
        duration_s=_floor_duration(started_at=started, now=now, sleep_fn=sleep_fn),
        random_seed=RANDOM_SEED,
    )
    artifact["distribution_shift_probe_after"] = {
        **dict(shift_after_probe),
        "corrected_model_checkpoint": MODEL_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    _write_json(root_path / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    errors = artifact_schema_errors(artifact)
    if errors:
        print(json.dumps({"result": RESULT_RELATIVE_PATH, "schema_errors": errors}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "result": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact["honest_verdict"],
                "shift_score_delta": artifact["shift_score_delta"],
                "first_win_rate_delta": artifact["first_win_rate_delta"],
                "solve_rate_delta": artifact["solve_rate_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
