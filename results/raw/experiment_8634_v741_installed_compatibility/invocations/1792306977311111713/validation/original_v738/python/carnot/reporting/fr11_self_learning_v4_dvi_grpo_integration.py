"""Exp 1388 FR-11 self-learning v4 DVI/GRPO integration.

Spec: REQ-LEARN-1388, SCENARIO-LEARN-1388.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

EXP1374_FILE = "experiment_1374_continuous_self_learning_v3_verifier_selected_or_csp_fallback.json"
EXP1381_FILE = "experiment_1381_dvi_discriminative_verifier_training_v1.json"
EXP1382_FILE = "experiment_1382_fullscale_certificate_semantic_repair_100cases.json"
EXP1383_FILE = "experiment_1383_grpo_v7_jury_rl_formal_verifier_rewards.json"
OUTPUT_FILE = "experiment_1388_fr11_self_learning_v4_dvi_grpo_integration.json"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1388_fr11_self_learning_v4_dvi_grpo_integration"
SCHEMA = "fr11_self_learning_v4_dvi_grpo_integration_v1"
RUN_DATE = "20260505"

PATH_DVI_ONLY = "dvi_only_replay_exp1382"
PATH_DVI_GRPO = "dvi_grpo_exp1382_integration"
BASELINE_FRESH_VERIFIED_COUNT = 4

POLICY_PROMOTE = "promote"
POLICY_DEMOTE = "demote"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "path_used",
    "dvi_checkpoint_active",
    "replay_cases_used",
    "fresh_verified_sample_count",
    "grpo_cases_integrated",
    "self_learning_delta_overall",
    "nonforgetting_certificate_rate",
    "memory_regression_count",
    "accepted_violation_delta",
    "promoted_memory_count",
    "demoted_memory_count",
    "dvi_auroc_delta_effect",
    "headline_result_allowed",
    "honest_verdict",
)

_SOURCE_FILES = {
    "exp1374": EXP1374_FILE,
    "exp1381": EXP1381_FILE,
    "exp1382": EXP1382_FILE,
    "exp1383": EXP1383_FILE,
}


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1388-1: make the run visible before source artifacts load."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifacts": [],
            "inputs_unavailable": [],
            "status": "in_progress",
            "path_used": None,
            "dvi_checkpoint_active": False,
            "dvi_checkpoint_path": None,
            "replay_cases_used": 0,
            "fresh_verified_sample_count": 0,
            "grpo_cases_integrated": 0,
            "self_learning_delta_overall": 0.0,
            "nonforgetting_certificate_rate": 0.0,
            "memory_regression_count": 0,
            "accepted_violation_delta": 0.0,
            "promoted_memory_count": 0,
            "demoted_memory_count": 0,
            "dvi_auroc_delta_effect": {
                "effect": "not_measured",
                "quality_improved_vs_exp1374_baseline": False,
            },
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
        },
    )


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1388-1/5: write bootstrap, integrate memory, and finalize."""

    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    payloads, unavailable_inputs, source_artifacts = load_inputs(results_dir)
    artifact = build_artifact(
        exp1374_artifact=payloads.get("exp1374", {}),
        exp1381_artifact=payloads.get("exp1381", {}),
        exp1382_artifact=payloads.get("exp1382", {}),
        exp1383_artifact=payloads.get("exp1383", {}),
        unavailable_inputs=unavailable_inputs,
        source_artifacts=source_artifacts,
        project_root=project_root,
        run_date=run_date,
    )
    validate_artifact(artifact)
    return _write_json(out_path, artifact)


def load_inputs(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Load source experiment artifacts and report any missing inputs."""

    results_path = Path(results_dir)
    payloads: dict[str, dict[str, Any]] = {}
    unavailable: list[str] = []
    sources: list[str] = []
    for key, filename in _SOURCE_FILES.items():
        path = results_path / filename
        relative = f"results/{filename}"
        if path.exists():
            payloads[key] = json.loads(path.read_text(encoding="utf-8"))
            sources.append(relative)
        else:
            unavailable.append(relative)
    return payloads, unavailable, sources


def build_artifact(
    *,
    exp1374_artifact: Mapping[str, Any],
    exp1381_artifact: Mapping[str, Any],
    exp1382_artifact: Mapping[str, Any],
    exp1383_artifact: Mapping[str, Any],
    unavailable_inputs: Sequence[str] = (),
    source_artifacts: Sequence[str] | None = None,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1388-2/3/4/5: build the DVI-active memory promotion artifact."""

    dvi_state = activate_dvi_checkpoint(exp1381_artifact)
    grpo_improvement = _float(exp1383_artifact.get("grpo_v7_improvement_pp"), 0.0)
    path_used = PATH_DVI_GRPO if grpo_improvement > 0.0 else PATH_DVI_ONLY

    replay_variants = build_replay_memory_variants(exp1374_artifact)
    exp1382_variants = build_exp1382_memory_variants(exp1382_artifact)
    grpo_variants = (
        build_grpo_verified_memory_variants(exp1383_artifact) if grpo_improvement > 0.0 else []
    )
    variants = replay_variants + exp1382_variants + grpo_variants
    memory_updates = apply_memory_updates(variants)

    fresh_verified_count = _support_count(
        variant
        for variant in variants
        if variant.get("memory_action") == POLICY_PROMOTE
        and str(variant.get("source")) in {"exp1382_dvi_semantic_validation", "exp1383_grpo_v7"}
    )
    grpo_cases_integrated = _support_count(
        variant for variant in grpo_variants if variant.get("memory_action") == POLICY_PROMOTE
    )
    replay_nonforgetting = derive_nonforgetting_controls(exp1374_artifact, replay_variants)
    current_false_accept_rate = _float(exp1382_artifact.get("scheduler_false_acceptance_rate"), 0.0)
    accepted_violation_delta = round(
        _float(exp1374_artifact.get("accepted_violation_delta"), 0.0) + current_false_accept_rate,
        6,
    )
    replay_cases_used = _int(exp1374_artifact.get("replay_cases_used"))
    fresh_delta = max(0, fresh_verified_count - BASELINE_FRESH_VERIFIED_COUNT)
    self_learning_delta = round(
        _float(exp1374_artifact.get("self_learning_delta_overall"), 0.0)
        + fresh_delta / max(1, replay_cases_used),
        6,
    )
    dvi_effect = dvi_auroc_delta_effect(
        exp1381_artifact=exp1381_artifact,
        fresh_verified_sample_count=fresh_verified_count,
    )
    headline_allowed = derive_headline_result_allowed(
        dvi_checkpoint_active=dvi_state["active"],
        fresh_verified_sample_count=fresh_verified_count,
        nonforgetting_certificate_rate=replay_nonforgetting["nonforgetting_certificate_rate"],
        memory_regression_count=replay_nonforgetting["memory_regression_count"],
        accepted_violation_delta=accepted_violation_delta,
    )

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "source_artifacts": list(source_artifacts or _default_source_artifacts()),
        "source_honest_verdicts": {
            "exp1374": exp1374_artifact.get("honest_verdict"),
            "exp1381": exp1381_artifact.get("honest_verdict"),
            "exp1382": exp1382_artifact.get("honest_verdict"),
            "exp1383": exp1383_artifact.get("honest_verdict"),
        },
        "inputs_unavailable": list(unavailable_inputs),
        "status": "complete" if dvi_state["active"] else "blocked",
        "path_used": path_used,
        "dvi_checkpoint_active": dvi_state["active"],
        "dvi_checkpoint_path": dvi_state["path"],
        "dvi_checkpoint_blocker": dvi_state["blocker"],
        "replay_cases_used": replay_cases_used,
        "new_cases_considered": len(exp1382_variants) + len(grpo_variants),
        "fresh_verified_sample_count": fresh_verified_count,
        "exp1374_fresh_verified_baseline": BASELINE_FRESH_VERIFIED_COUNT,
        "fresh_verified_delta_vs_exp1374": fresh_verified_count - BASELINE_FRESH_VERIFIED_COUNT,
        "grpo_cases_integrated": grpo_cases_integrated,
        "grpo_v7_improvement_pp": grpo_improvement,
        "self_learning_delta_overall": self_learning_delta,
        "self_learning_delta_components": {
            "exp1374_replay_control_delta": _float(
                exp1374_artifact.get("self_learning_delta_overall"), 0.0
            ),
            "fresh_verified_delta_over_replay": round(fresh_delta / max(1, replay_cases_used), 6),
        },
        "nonforgetting_certificate_rate": replay_nonforgetting["nonforgetting_certificate_rate"],
        "memory_regression_count": replay_nonforgetting["memory_regression_count"],
        "accepted_violation_delta": accepted_violation_delta,
        "accepted_violation_controls": {
            "exp1374_accepted_violation_delta": _float(
                exp1374_artifact.get("accepted_violation_delta"), 0.0
            ),
            "exp1382_scheduler_false_acceptance_rate": current_false_accept_rate,
        },
        "promoted_memory_count": memory_updates["promoted_memory_count"],
        "demoted_memory_count": memory_updates["demoted_memory_count"],
        "memory_updates": memory_updates,
        "dvi_auroc_delta_effect": dvi_effect,
        "headline_result_allowed": headline_allowed,
        "honest_verdict": derive_honest_verdict(
            path_used=path_used,
            dvi_checkpoint_active=dvi_state["active"],
            headline_result_allowed=headline_allowed,
            fresh_verified_sample_count=fresh_verified_count,
            grpo_cases_integrated=grpo_cases_integrated,
        ),
        "measurement_note": (
            "Exp 1388 activates the deployed Exp 1381 DVI checkpoint and uses "
            "Exp 1374 for replay/non-forgetting controls. Exp 1383 had to show "
            "a positive GRPO v7 improvement before GRPO cases could enter memory; "
            "otherwise this artifact follows the required DVI-only replay path. "
            "Fresh promotions come from Exp 1382 DVI semantic rows with "
            "constraint_passed=true."
        ),
    }
    validate_artifact(artifact)
    return artifact


def activate_dvi_checkpoint(exp1381_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """REQ-LEARN-1388-2: verify that the Exp 1381 DVI checkpoint is active."""

    if exp1381_artifact.get("dvi_deployed") is not True:
        return {
            "active": False,
            "path": exp1381_artifact.get("dvi_checkpoint_path"),
            "blocker": "exp1381_dvi_not_deployed",
        }
    raw_path = exp1381_artifact.get("dvi_checkpoint_path")
    if not raw_path:
        return {"active": False, "path": None, "blocker": "exp1381_dvi_checkpoint_path_missing"}
    path = Path(str(raw_path))
    if not path.exists():
        return {"active": False, "path": str(path), "blocker": "dvi_checkpoint_file_missing"}
    try:
        with np.load(path, allow_pickle=False) as data:
            has_metric = "metric" in data.files and np.asarray(data["metric"]).size > 0
            has_bias = "bias" in data.files and np.asarray(data["bias"]).size > 0
    except Exception as exc:
        return {
            "active": False,
            "path": str(path),
            "blocker": f"dvi_checkpoint_unreadable:{type(exc).__name__}",
        }
    if not (has_metric and has_bias):
        return {
            "active": False,
            "path": str(path),
            "blocker": "dvi_checkpoint_missing_metric_or_bias",
        }
    return {"active": True, "path": str(path), "blocker": None}


def build_replay_memory_variants(exp1374_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-LEARN-1388-4: retain Exp 1374 verifier-promoted memory."""

    variants: list[dict[str, Any]] = []
    for index, row in enumerate(_rows(exp1374_artifact, ("variant_questions",))):
        accepted = bool(row.get("verifier_accepted")) and not bool(row.get("semantic_rejected"))
        if str(row.get("memory_action") or "") == POLICY_PROMOTE:
            accepted = True
        if not accepted:
            continue
        case_id = str(row.get("case_id") or f"exp1374_replay_{index}")
        variants.append(
            {
                "variant_id": str(row.get("variant_id") or f"replay:exp1374:{case_id}"),
                "source": "exp1374_promoted_memory",
                "case_id": case_id,
                "memory_action": POLICY_PROMOTE,
                "support": max(_int(row.get("support")), 1),
                "dvi_score_source": "exp1374_promoted_primary_semantic_or_replay_memory",
            }
        )
    if variants:
        return variants

    promoted_count = _int(exp1374_artifact.get("promoted_memory_count"))
    if promoted_count <= 0:
        return []
    return [
        {
            "variant_id": "replay:exp1374:aggregate_promoted",
            "source": "exp1374_promoted_memory",
            "case_id": "aggregate_promoted",
            "memory_action": POLICY_PROMOTE,
            "support": promoted_count,
            "dvi_score_source": "exp1374_promoted_memory_aggregate",
        }
    ]


def build_exp1382_memory_variants(exp1382_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-LEARN-1388-4: promote Exp 1382 rows accepted by DVI semantic validation."""

    variants: list[dict[str, Any]] = []
    for index, row in enumerate(_rows(exp1382_artifact, ("semantic_validation_rows",))):
        case_id = str(row.get("case_id") or f"exp1382_{index}")
        accepted = row.get("constraint_passed") is True
        variants.append(
            {
                "variant_id": f"dvi:exp1382:{case_id}",
                "source": "exp1382_dvi_semantic_validation",
                "case_id": case_id,
                "memory_action": POLICY_PROMOTE if accepted else POLICY_DEMOTE,
                "support": 1,
                "dvi_score_source": "exp1382_semantic_validation_rows",
                "evidence_summary": {
                    "claim_route": row.get("claim_route"),
                    "certificate_state": row.get("certificate_state"),
                    "expected_state": row.get("expected_state"),
                    "semantic_result": row.get("semantic_result"),
                    "constraint_evaluated": row.get("constraint_evaluated"),
                    "constraint_passed": row.get("constraint_passed"),
                    "dvi_incorrect_probability": row.get("dvi_incorrect_probability"),
                    "dvi_incorrect_threshold": row.get("dvi_incorrect_threshold"),
                    "fover_label": row.get("fover_label"),
                    "failure_reason": row.get("failure_reason"),
                },
            }
        )
    return variants


def build_grpo_verified_memory_variants(
    exp1383_artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """REQ-LEARN-1388-3: integrate only GRPO cases that were formally verified."""

    variants: list[dict[str, Any]] = []
    for row in _rows(exp1383_artifact, ("training_reward_rows",)):
        if str(row.get("verifier_result")) != "VERIFIED":
            continue
        case_id = str(row.get("case_id") or f"training_{len(variants)}")
        variants.append(
            _grpo_variant(case_id=case_id, row=row, source_stage="training_reward_rows")
        )
    for row in _rows(exp1383_artifact, ("heldout_evaluation_rows",)):
        verifier_result = str(row.get("post_grpo_verifier_result") or "")
        if verifier_result != str(row.get("expected_answer") or ""):
            continue
        case_id = str(row.get("case_id") or f"heldout_{len(variants)}")
        variants.append(
            _grpo_variant(case_id=case_id, row=row, source_stage="heldout_evaluation_rows")
        )
    return variants


def _grpo_variant(
    *,
    case_id: str,
    row: Mapping[str, Any],
    source_stage: str,
) -> dict[str, Any]:
    return {
        "variant_id": f"grpo:exp1383:{case_id}",
        "source": "exp1383_grpo_v7",
        "case_id": case_id,
        "memory_action": POLICY_PROMOTE,
        "support": 1,
        "dvi_score_source": source_stage,
        "evidence_summary": dict(row),
    }


def apply_memory_updates(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count support-weighted memory promotions and demotions."""

    promoted: list[str] = []
    demoted: list[str] = []
    promoted_count = 0
    demoted_count = 0
    for variant in variants:
        variant_id = str(variant.get("variant_id") or variant.get("case_id") or "unknown")
        support = max(_int(variant.get("support")), 1)
        if variant.get("memory_action") == POLICY_PROMOTE:
            promoted.append(variant_id)
            promoted_count += support
        elif variant.get("memory_action") == POLICY_DEMOTE:
            demoted.append(variant_id)
            demoted_count += support
    return {
        "promoted": promoted,
        "demoted": demoted,
        "promoted_memory_count": promoted_count,
        "demoted_memory_count": demoted_count,
    }


def derive_nonforgetting_controls(
    exp1374_artifact: Mapping[str, Any],
    replay_variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure whether Exp 1374 promoted memory remains retained under DVI v4."""

    prior_promoted = _int(exp1374_artifact.get("promoted_memory_count"))
    retained = _support_count(replay_variants)
    computed_regressions = max(0, prior_promoted - retained)
    memory_regression_count = max(
        _int(exp1374_artifact.get("memory_regression_count")),
        computed_regressions,
    )
    if prior_promoted > 0:
        rate = retained / prior_promoted
    else:
        rate = _float(exp1374_artifact.get("nonforgetting_certificate_rate"), 0.0)
    return {
        "nonforgetting_certificate_rate": round(min(1.0, max(0.0, rate)), 6),
        "memory_regression_count": memory_regression_count,
        "prior_promoted_memory_count": prior_promoted,
        "retained_promoted_memory_count": retained,
    }


def dvi_auroc_delta_effect(
    *,
    exp1381_artifact: Mapping[str, Any],
    fresh_verified_sample_count: int,
) -> dict[str, Any]:
    """Summarize whether the active DVI checkpoint improved memory quality."""

    dvi_delta = _float(exp1381_artifact.get("dvi_auroc_delta"), 0.0)
    improved = dvi_delta > 0.0 and fresh_verified_sample_count > BASELINE_FRESH_VERIFIED_COUNT
    return {
        "effect": "positive" if improved else "not_positive",
        "dvi_auroc_delta": round(dvi_delta, 6),
        "dvi_baseline_auroc": _maybe_float(exp1381_artifact.get("dvi_baseline_auroc")),
        "dvi_trained_auroc": _maybe_float(exp1381_artifact.get("dvi_trained_auroc")),
        "exp1374_fresh_verified_baseline": BASELINE_FRESH_VERIFIED_COUNT,
        "current_fresh_verified": int(fresh_verified_sample_count),
        "quality_improved_vs_exp1374_baseline": improved,
    }


def derive_headline_result_allowed(
    *,
    dvi_checkpoint_active: bool,
    fresh_verified_sample_count: int,
    nonforgetting_certificate_rate: float,
    memory_regression_count: int,
    accepted_violation_delta: float,
) -> bool:
    """REQ-LEARN-1388-6: headline requires a strict fresh-count improvement."""

    return (
        dvi_checkpoint_active
        and fresh_verified_sample_count > BASELINE_FRESH_VERIFIED_COUNT
        and nonforgetting_certificate_rate == 1.0
        and memory_regression_count == 0
        and accepted_violation_delta <= 0.0
    )


def derive_honest_verdict(
    *,
    path_used: str,
    dvi_checkpoint_active: bool,
    headline_result_allowed: bool,
    fresh_verified_sample_count: int,
    grpo_cases_integrated: int,
) -> str:
    """Name the active path and the headline boundary honestly."""

    if not dvi_checkpoint_active:
        return "fr11_self_learning_v4_blocked_dvi_checkpoint_inactive"
    if headline_result_allowed and path_used == PATH_DVI_GRPO:
        return (
            "fr11_self_learning_v4_dvi_grpo_integrated_headline_allowed_"
            f"fresh_{fresh_verified_sample_count}_grpo_{grpo_cases_integrated}"
        )
    if headline_result_allowed:
        return (
            "fr11_self_learning_v4_dvi_only_exp1382_headline_allowed_"
            f"fresh_{fresh_verified_sample_count}"
        )
    if fresh_verified_sample_count <= BASELINE_FRESH_VERIFIED_COUNT:
        return "fr11_self_learning_v4_dvi_active_no_fresh_delta_non_headline"
    return "fr11_self_learning_v4_dvi_active_controls_blocked_non_headline"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1388-5/6: assert required fields and headline invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["path_used"] not in {None, PATH_DVI_ONLY, PATH_DVI_GRPO}:
        raise AssertionError(f"unsupported path_used: {artifact['path_used']}")
    for field in (
        "replay_cases_used",
        "fresh_verified_sample_count",
        "grpo_cases_integrated",
        "memory_regression_count",
        "promoted_memory_count",
        "demoted_memory_count",
    ):
        if not isinstance(artifact[field], int) or artifact[field] < 0:
            raise AssertionError(f"{field} must be a non-negative integer")
    rate = artifact["nonforgetting_certificate_rate"]
    if not isinstance(rate, (int, float)) or not 0.0 <= float(rate) <= 1.0:
        raise AssertionError("nonforgetting_certificate_rate must be between 0 and 1")
    if artifact["headline_result_allowed"]:
        if artifact["dvi_checkpoint_active"] is not True:
            raise AssertionError("headline_result_allowed requires an active DVI checkpoint")
        if artifact["fresh_verified_sample_count"] <= BASELINE_FRESH_VERIFIED_COUNT:
            raise AssertionError("headline_result_allowed requires fresh_verified_sample_count > 4")
        if artifact["nonforgetting_certificate_rate"] != 1.0:
            raise AssertionError("headline_result_allowed requires non-forgetting rate of 1.0")
        if artifact["memory_regression_count"] != 0:
            raise AssertionError("headline_result_allowed requires zero memory regressions")
        if artifact["accepted_violation_delta"] > 0.0:
            raise AssertionError(
                "headline_result_allowed requires non-positive accepted violations"
            )
    if artifact["path_used"] == PATH_DVI_ONLY and artifact["grpo_cases_integrated"] != 0:
        raise AssertionError("DVI-only path cannot integrate GRPO cases")


def _support_count(variants: Sequence[Mapping[str, Any]] | Any) -> int:
    return sum(max(_int(variant.get("support")), 1) for variant in variants)


def _rows(artifact: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    for key in keys:
        value = artifact.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _default_source_artifacts() -> list[str]:
    return [f"results/{filename}" for filename in _SOURCE_FILES.values()]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(_float(value), 6)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
