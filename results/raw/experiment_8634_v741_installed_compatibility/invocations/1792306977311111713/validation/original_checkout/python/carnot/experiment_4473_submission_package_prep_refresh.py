"""Exp 4473: refresh the operator-only ARC replay submission package.

Spec refs: REQ-REPORT-4473, SCENARIO-REPORT-4473.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from carnot import experiment_4460_submission_package_prep as base


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4473_submission_package_prep_refresh.json"
OPERATOR_NOTE_RELATIVE_PATH = "docs/research-notes/arc3-submission-package-4473-operator-note.md"
REGISTRY_RELATIVE_PATH = base.REGISTRY_RELATIVE_PATH
PRIOR_SUBMISSION_RELATIVE_PATH = base.PRIOR_SUBMISSION_RELATIVE_PATH
PRIOR_PACKAGE_412_RELATIVE_PATH = base.RESULT_RELATIVE_PATH
PRIOR_PACKAGE_412_LEVELS = 39
PRIOR_SUBMITTED_BASELINE_LEVELS = 13
RANDOM_SEED = 4473
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- re-validates cached reproduce "
    "sequences against the offline env (1s floor); never None, never live_llm_inference"
)
BLOCKED_INFERENCE_SUBSTRATE = base.BLOCKED_INFERENCE_SUBSTRATE
TERMINAL_PREFIXES = base.TERMINAL_PREFIXES

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "submission_package_ready",
    "total_reproduced_levels_in_package",
    "grew_vs_412",
    "prior_submitted_baseline_levels",
    "beats_prior_baseline",
    "per_game_replay_validation",
    "submitted_to_leaderboard",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {"principle": "terminal-prefixed"},
    "inference_substrate": {"principle": INFERENCE_SUBSTRATE},
    "submission_package_ready": {
        "principle": (
            "bare bool: TRUE if the package is ready for the OPERATOR to submit; "
            "the task itself NEVER submits (Operator-Only External Publication)"
        )
    },
    "total_reproduced_levels_in_package": {
        "principle": (
            "bare int: env-match-validated reproduced levels in the package "
            "(target > 39 after the .413 banks)"
        )
    },
    "grew_vs_412": {
        "principle": "bare bool: total > 39 -- the .413 package added levels over .412"
    },
    "prior_submitted_baseline_levels": {
        "principle": "bare int = 13; the baseline the package must beat"
    },
    "beats_prior_baseline": {
        "principle": "bare bool: total_reproduced_levels_in_package > 13"
    },
    "per_game_replay_validation": {
        "principle": (
            "list of {game, replays_ok, reproduced_levels, env_matched} -- the "
            "audit trail; quarantined games excluded from the count"
        )
    },
    "submitted_to_leaderboard": {
        "principle": "bare bool MUST be false -- the task never submits"
    },
    "verifier_is_oracle": {
        "principle": "true: execution-grounded reproduction re-validation"
    },
    "random_seed": {"principle": "determinism"},
    "reproducibility_checksum": {
        "principle": "content hash of the package manifest"
    },
}

ReproduceEntryFn = base.ReproduceEntryFn


def _prior_package_412_levels(root: Path) -> int:
    artifact = base._load_json(root / PRIOR_PACKAGE_412_RELATIVE_PATH)
    return base._as_int(artifact.get("total_reproduced_levels_in_package")) or PRIOR_PACKAGE_412_LEVELS


def _honest_verdict(*, ready: bool, total: int, baseline: int, grew: bool, quarantined_count: int) -> str:
    if ready:
        growth = "grew_vs_412" if grew else "not_grown_vs_412"
        return f"success: submission_package_ready_{total}_levels_beats_{baseline}_{growth}_quarantined_{quarantined_count}"
    return f"complete: submission_package_not_ready_{total}_levels_vs_{baseline}_quarantined_{quarantined_count}"


def resolve_replay_plan(entry: Mapping[str, Any], root: Path = REPO_ROOT) -> base.ReplayPlan:  # pragma: no cover
    """SCENARIO-REPORT-4473: find current-depth cached labels for .413 rows."""

    game = str(entry.get("game") or "")
    claimed = base._as_int(entry.get("levels_reproduced"))
    if game == "sc25":
        from carnot import experiment_4468_bank_sc25_provisional_levels as exp4468

        level = claimed if claimed in exp4468.SC25_PLANS_BY_LEVEL else max(exp4468.SC25_PLANS_BY_LEVEL)
        return base.ReplayPlan(
            game,
            [str(label) for label in exp4468.SC25_PLANS_BY_LEVEL[level]],
            exp4468.RESULT_RELATIVE_PATH,
            exp4468.apply_sc25_label,
            warmup_label="warmup",
        )
    if game == "dc22":
        from carnot import experiment_4467_solve_dc22_cegis_nocov as exp4467

        return base.ReplayPlan(
            game,
            [str(label) for label in exp4467.DC22_L1_SOLUTION],
            exp4467.RESULT_RELATIVE_PATH,
            exp4467.apply_dc22_label,
        )
    if game == "sb26":
        from carnot import experiment_4470_color_match_slot_operator_solve_sb26 as exp4470

        artifact = base._load_json(root / exp4470.RESULT_RELATIVE_PATH)
        labels = artifact.get("solution_labels") if isinstance(artifact.get("solution_labels"), list) else []
        return base.ReplayPlan(
            game,
            [str(label) for label in labels],
            exp4470.RESULT_RELATIVE_PATH,
            exp4470.apply_sb26_label,
        )
    return base.resolve_replay_plan(entry, root)


def reproduce_registry_entry(entry: Mapping[str, Any], root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    from carnot.agentic import arc_solver_kit

    game = str(entry.get("game") or "")
    claimed = base._as_int(entry.get("levels_reproduced"))
    plan = resolve_replay_plan(entry, root)
    if not plan.labels:
        return {
            "game": game,
            "claimed_level": claimed,
            "reached_level": 0,
            "reproduced": False,
            "source": plan.source,
            "action_sequence": [],
            "action_count": 0,
            "gate": "missing_cached_replay_plan",
        }
    result = dict(
        arc_solver_kit.reproduce(
            game,
            plan.labels,
            plan.apply_fn,
            warmup_label=plan.warmup_label,
            claimed_level=claimed,
        )
    )
    result["source"] = plan.source
    result["action_sequence"] = list(plan.labels)
    result["action_count"] = len(plan.labels)
    result["gate"] = "arc_solver_kit.reproduce"
    if plan.warmup_label is not None:
        result["warmup_label"] = plan.warmup_label
    return result


def compute_reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return base._sha256(
        {
            "package_manifest": artifact.get("package_manifest", []),
            "total_reproduced_levels_in_package": artifact.get("total_reproduced_levels_in_package"),
            "grew_vs_412": artifact.get("grew_vs_412"),
            "prior_submitted_baseline_levels": artifact.get("prior_submitted_baseline_levels"),
            "submitted_to_leaderboard": artifact.get("submitted_to_leaderboard"),
            "random_seed": artifact.get("random_seed"),
        }
    )


def _blocked_artifact(
    *,
    reason: str,
    registry: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    started_at: float,
    ended_at: float,
    prior_package_412_levels: int,
) -> dict[str, Any]:
    baseline = base._as_int(registry.get("prior_submitted_baseline_levels")) or PRIOR_SUBMITTED_BASELINE_LEVELS
    artifact: dict[str, Any] = {
        "experiment": "experiment_4473_submission_package_prep_refresh",
        "schema": "carnot.exp4473.submission_package_prep_refresh.v1",
        "honest_verdict": f"complete: blocked_{reason}",
        "inference_substrate": BLOCKED_INFERENCE_SUBSTRATE,
        "submission_package_ready": False,
        "total_reproduced_levels_in_package": 0,
        "prior_package_412_levels": prior_package_412_levels,
        "grew_vs_412": False,
        "prior_submitted_baseline_levels": baseline,
        "beats_prior_baseline": False,
        "per_game_replay_validation": [],
        "package_manifest": [],
        "quarantined_games": [],
        "operator_checklist": [],
        "operator_note_path": OPERATOR_NOTE_RELATIVE_PATH,
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "duration_s": base._duration(started_at, ended_at),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-REPORT-4473", "SCENARIO-REPORT-4473"],
        "no_3090_inference": True,
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    registry: Mapping[str, Any],
    validation_rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    started_at: float,
    ended_at: float,
    prior_package_412_levels: int,
) -> dict[str, Any]:
    manifest = base._package_manifest(validation_rows)
    total = sum(base._as_int(row.get("levels")) for row in manifest)
    baseline = base._as_int(registry.get("prior_submitted_baseline_levels")) or PRIOR_SUBMITTED_BASELINE_LEVELS
    beats_baseline = total > baseline
    grew = total > prior_package_412_levels
    quarantined = [str(row.get("game")) for row in validation_rows if row.get("quarantined")]
    ready = bool(beats_baseline and manifest)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4473_submission_package_prep_refresh",
        "schema": "carnot.exp4473.submission_package_prep_refresh.v1",
        "honest_verdict": _honest_verdict(
            ready=ready,
            total=total,
            baseline=baseline,
            grew=grew,
            quarantined_count=len(quarantined),
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "submission_package_ready": ready,
        "total_reproduced_levels_in_package": total,
        "prior_package_412_levels": prior_package_412_levels,
        "grew_vs_412": grew,
        "prior_submitted_baseline_levels": baseline,
        "beats_prior_baseline": beats_baseline,
        "per_game_replay_validation": [dict(row) for row in validation_rows],
        "package_manifest": manifest,
        "quarantined_games": quarantined,
        "operator_checklist": _operator_checklist(total=total, baseline=baseline, manifest=manifest),
        "operator_note_path": OPERATOR_NOTE_RELATIVE_PATH,
        "submitted_to_leaderboard": False,
        "verifier_is_oracle": True,
        "random_seed": RANDOM_SEED,
        "duration_s": base._duration(started_at, ended_at),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-REPORT-4473", "SCENARIO-REPORT-4473"],
        "no_3090_inference": True,
        "result_path": RESULT_RELATIVE_PATH,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(artifact)
    return artifact


def _operator_checklist(*, total: int, baseline: int, manifest: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        "Review this refreshed JSON artifact and the package_manifest rows before submitting.",
        f"Confirm total_reproduced_levels_in_package={total} is greater than prior baseline {baseline}.",
        f"Confirm grew_vs_412 reflects whether total_reproduced_levels_in_package>{PRIOR_PACKAGE_412_LEVELS}.",
        "Run scripts/arc3_live_submit.py only as the operator; this prep task did not submit.",
        f"Package contains {len(manifest)} replayable games with cached action sequences.",
        "After any operator live validation, record the resulting scorecard separately.",
    ]


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing {field}")
    if not base._terminal_prefixed(artifact.get("honest_verdict")):
        errors.append("honest_verdict must be terminal-prefixed")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE and artifact.get("inference_substrate") != BLOCKED_INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal refresh substrate")
    if type(artifact.get("submission_package_ready")) is not bool:
        errors.append("submission_package_ready must be bare bool")
    if type(artifact.get("total_reproduced_levels_in_package")) is not int:
        errors.append("total_reproduced_levels_in_package must be bare int")
    if type(artifact.get("grew_vs_412")) is not bool:
        errors.append("grew_vs_412 must be bare bool")
    if type(artifact.get("prior_submitted_baseline_levels")) is not int:
        errors.append("prior_submitted_baseline_levels must be bare int")
    if type(artifact.get("beats_prior_baseline")) is not bool:
        errors.append("beats_prior_baseline must be bare bool")
    if not isinstance(artifact.get("per_game_replay_validation"), list):
        errors.append("per_game_replay_validation must be list")
    if artifact.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard must be false")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be bare int")
    if not base._checksum_is_hex(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be sha256 hex")
    total = artifact.get("total_reproduced_levels_in_package")
    if type(total) is int and type(artifact.get("grew_vs_412")) is bool:
        expected_growth = total > base._as_int(artifact.get("prior_package_412_levels")) if "prior_package_412_levels" in artifact else total > PRIOR_PACKAGE_412_LEVELS
        if artifact.get("grew_vs_412") is not expected_growth:
            errors.append("grew_vs_412 inconsistent with total and .412 baseline")
    if artifact.get("submission_package_ready") is True:
        if artifact.get("beats_prior_baseline") is not True:
            errors.append("ready package must beat prior baseline")
        if not artifact.get("package_manifest"):
            errors.append("ready package must include package_manifest rows")
        if artifact.get("submitted_to_leaderboard") is not False:
            errors.append("ready package must not submit")
    return errors


def write_artifact(root: Path, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def write_operator_note(root: Path, artifact: Mapping[str, Any]) -> Path:
    path = root / OPERATOR_NOTE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = artifact.get("package_manifest") if isinstance(artifact.get("package_manifest"), list) else []
    lines = [
        "# ARC-AGI-3 Operator Submission Package Refresh (Exp 4473)",
        "",
        f"- Artifact: `{RESULT_RELATIVE_PATH}`",
        f"- Ready for operator submission: `{artifact.get('submission_package_ready')}`",
        f"- Revalidated package levels: `{artifact.get('total_reproduced_levels_in_package')}`",
        f"- Grew vs Exp 4460/.412: `{artifact.get('grew_vs_412')}`",
        f"- Prior submitted baseline: `{artifact.get('prior_submitted_baseline_levels')}`",
        f"- Submitted by this task: `{artifact.get('submitted_to_leaderboard')}`",
        "",
        "Operator checklist:",
    ]
    for item in artifact.get("operator_checklist", []):
        lines.append(f"- {item}")
    lines.extend(["", "Package manifest:"])
    for row in manifest:
        lines.append(
            f"- {row.get('game')}: L{row.get('levels')}, actions={row.get('action_count')}, "
            f"env_match_basis={row.get('env_match_basis')}, source={row.get('source')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(
    root: Path = REPO_ROOT,
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
    reproduce_entry_fn: ReproduceEntryFn = reproduce_registry_entry,
    now: Callable[[], float] = time.perf_counter,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """SCENARIO-REPORT-4473: validate cached .413 replays and write the refreshed package."""

    started = now()
    registry = base.load_registry(root)
    prior_package_412_levels = _prior_package_412_levels(root)
    checked = dict(preconditions_checked or base.precondition_probe(root))
    miss = base.first_precondition_miss(checked)
    if miss is not None:
        artifact = _blocked_artifact(
            reason=miss,
            registry=registry,
            preconditions_checked=checked,
            started_at=started,
            ended_at=now(),
            prior_package_412_levels=prior_package_412_levels,
        )
        write_artifact(root, artifact)
        write_operator_note(root, artifact)
        return artifact

    validation_rows = base.validate_registry_replays(
        root,
        registry=registry,
        reproduce_entry_fn=reproduce_entry_fn,
    )
    ended = base._floor_end_time(started_at=started, now=now, sleep_fn=sleep_fn)
    artifact = build_artifact(
        registry=registry,
        validation_rows=validation_rows,
        preconditions_checked=checked,
        started_at=started,
        ended_at=ended,
        prior_package_412_levels=prior_package_412_levels,
    )
    write_artifact(root, artifact)
    write_operator_note(root, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run(REPO_ROOT)
    print(REPO_ROOT / RESULT_RELATIVE_PATH)
    print(artifact["honest_verdict"])
    print(f"submitted_to_leaderboard={artifact['submitted_to_leaderboard']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
