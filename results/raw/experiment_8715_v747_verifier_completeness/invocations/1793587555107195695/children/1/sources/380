"""Exp 3045 FR-11 governed self-learning boundary.

This module is a governance artifact builder, not a learning loop. It reads
the terminal Exp 3032 and Exp 3033 JSON artifacts, carries forward only the
controller-side evidence they actually provide, and writes the protocol that
Exp 3046 must satisfy before any stronger FR-11 self-learning claim is allowed.

Spec refs: REQ-LEARN-3045, SCENARIO-LEARN-3045,
SCENARIO-LEARN-3045-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3045_fr11_governed_self_learning_boundary_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.governed_self_learning_boundary.v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME

EXP3032_ARTIFACT_REL_PATH = Path("results/experiment_3032_fr11_heldout_dvi_replay_v2.json")
EXP3033_ARTIFACT_REL_PATH = Path(
    "results/experiment_3033_fr11_nonforgetting_negative_control_stress_v1.json"
)

TERMINAL_SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
BLOCKED_PREFIXES = ("blocked:", "blocked_")

EDIT_TARGET_NAMES = {
    "controller_weights",
    "trace_memory",
    "validator_thresholds",
    "kan_locality_anchors",
    "model_weights",
}
REQUIRED_METRIC_NAMES = {
    "family_holdout_delta",
    "prior_retention_delta",
    "no_feedback_delta",
    "shuffled_control_delta",
    "contradiction_graph_size_rate",
    "rollback_count",
    "delayed_regression_delta",
    "source_trace_completeness",
}
NON_PROMOTION_NAMES = {
    "tautology",
    "self_confirming_labels",
    "family_leakage",
    "missing_negative_controls",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_governance_ready",
        "allowed_edit_targets",
        "forbidden_claims",
        "required_metrics",
        "non_promotion_criteria",
        "prior_evidence_summary",
        "continuous_self_learning_scope",
        "inference_substrate",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for the governance artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3032_artifact_path: Path | None = None
    exp3033_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_exp3032_artifact_path(self) -> Path:
        return self.exp3032_artifact_path or self.repo_root / EXP3032_ARTIFACT_REL_PATH

    def resolved_exp3033_artifact_path(self) -> Path:
        return self.exp3033_artifact_path or self.repo_root / EXP3033_ARTIFACT_REL_PATH


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3045 governance artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    exp3032 = _load_json(active.resolved_exp3032_artifact_path())
    exp3033 = _load_json(active.resolved_exp3033_artifact_path())
    blocker = precondition_blocker(exp3032, exp3033)
    duration_s = _round(active.clock() - started)

    if blocker is not None:
        artifact = _blocked_artifact(active, exp3032, exp3033, blocker, duration_s)
    else:
        artifact = _complete_artifact(active, exp3032, exp3033, duration_s)

    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def precondition_blocker(exp3032: Mapping[str, Any], exp3033: Mapping[str, Any]) -> str | None:
    """Return the first source-evidence blocker, or None when governance may proceed."""

    exp3032_blocker = _source_blocker(
        "exp3032",
        exp3032,
        ready_field="fr11_heldout_replay_ready",
        promotable_field=None,
    )
    if exp3032_blocker is not None:
        return exp3032_blocker
    return _source_blocker(
        "exp3033",
        exp3033,
        ready_field="fr11_nonforgetting_stress_ready",
        promotable_field="fr11_self_learning_promotable",
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate that the artifact is complete enough to gate Exp 3046."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")

    ready = bool(artifact["fr11_governance_ready"])
    honest_verdict = str(artifact["honest_verdict"])
    if ready and not honest_verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if not ready and not honest_verdict.startswith(BLOCKED_PREFIXES):
        raise ValueError("blocked artifacts must use a blocked_ prefix")

    edit_targets = _rows_by_name(artifact["allowed_edit_targets"], "allowed_edit_targets")
    if set(edit_targets) != EDIT_TARGET_NAMES:
        raise ValueError("allowed_edit_targets must enumerate every governed edit target")
    model_target = edit_targets["model_weights"]
    if model_target.get("scope") != "out_of_scope":
        raise ValueError("model_weights must remain out of scope")
    if model_target.get("requires_actual_training_experiment") is not True:
        raise ValueError("model_weights must require an actual training experiment")

    metric_rows = _rows_by_name(artifact["required_metrics"], "required_metrics")
    if set(metric_rows) != REQUIRED_METRIC_NAMES:
        raise ValueError("required_metrics must enumerate every Exp 3046 metric")
    if any(row.get("required_for_exp3046") is not True for row in metric_rows.values()):
        raise ValueError("required_metrics rows must be required_for_exp3046")

    criteria_rows = _rows_by_name(artifact["non_promotion_criteria"], "non_promotion_criteria")
    if set(criteria_rows) != NON_PROMOTION_NAMES:
        raise ValueError("non_promotion_criteria must predeclare every failure mode")
    if any(row.get("automatic") is not True for row in criteria_rows.values()):
        raise ValueError("non_promotion_criteria rows must be automatic")

    forbidden_claims = artifact["forbidden_claims"]
    if not isinstance(forbidden_claims, list) or len(forbidden_claims) < 4:
        raise ValueError("forbidden_claims must list the claim boundary")
    if not any("model-weight learning" in str(claim) for claim in forbidden_claims):
        raise ValueError("forbidden_claims must prohibit model-weight learning claims")

    substrate = artifact["inference_substrate"]
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if substrate.get("cached_artifacts_only") is not True:
        raise ValueError("inference_substrate must declare cached_artifacts_only")
    if substrate.get("live_llm_inference") is not False:
        raise ValueError("inference_substrate must forbid live LLM inference")
    if substrate.get("model_weight_training") is not False:
        raise ValueError("inference_substrate must forbid model weight training")
    if substrate.get("model_weight_mutation") is not False:
        raise ValueError("inference_substrate must forbid model weight mutation")

    summary = artifact["prior_evidence_summary"]
    if not isinstance(summary, Mapping) or "exp3032" not in summary or "exp3033" not in summary:
        raise ValueError("prior_evidence_summary must include exp3032 and exp3033")
    if ready and not (summary["exp3032"].get("ready") and summary["exp3033"].get("ready")):
        raise ValueError("prior_evidence_summary must show both sources ready")

    scope = str(artifact["continuous_self_learning_scope"])
    if "controller-only" not in scope or "model weights" not in scope:
        raise ValueError("continuous_self_learning_scope must state the controller-only boundary")


def allowed_edit_targets() -> list[JsonDict]:
    """Return the explicit edit-right boundary for downstream self-learning."""

    return [
        {
            "name": "controller_weights",
            "scope": "allowed_controller_side",
            "principle": "bounded verifier-feedback controllers may update inspectable weights",
            "requires_actual_training_experiment": False,
        },
        {
            "name": "trace_memory",
            "scope": "allowed_controller_side",
            "principle": "source-traced memories may be added only with replay evidence",
            "requires_actual_training_experiment": False,
        },
        {
            "name": "validator_thresholds",
            "scope": "allowed_controller_side",
            "principle": "threshold changes require held-out and negative-control deltas",
            "requires_actual_training_experiment": False,
        },
        {
            "name": "kan_locality_anchors",
            "scope": "allowed_controller_side",
            "principle": "locality anchors may govern update locality without native KAN retraining",
            "requires_actual_training_experiment": False,
        },
        {
            "name": "model_weights",
            "scope": "out_of_scope",
            "principle": "model-weight learning must be separated from controller-side learning",
            "requires_actual_training_experiment": True,
        },
    ]


def required_metrics() -> list[JsonDict]:
    """Return the exact Exp 3046 metric gate list."""

    return [
        _metric("family_holdout_delta", "held-out family utility must improve after update"),
        _metric("prior_retention_delta", "previous exact traces must not regress"),
        _metric("no_feedback_delta", "replay without feedback must not improve"),
        _metric("shuffled_control_delta", "shuffled labels must not improve the controller"),
        _metric(
            "contradiction_graph_size_rate", "graph size and contradiction rate must be logged"
        ),
        _metric("rollback_count", "all automatic rollbacks must be counted"),
        _metric("delayed_regression_delta", "delayed replay must expose late regressions"),
        _metric("source_trace_completeness", "every promoted update needs source-trace coverage"),
    ]


def non_promotion_criteria() -> list[JsonDict]:
    """Return failure modes that automatically deny FR-11 promotion."""

    return [
        {
            "name": "tautology",
            "automatic": True,
            "criterion": "deny promotion if the scoring channel can grade its own update path",
        },
        {
            "name": "self_confirming_labels",
            "automatic": True,
            "criterion": "deny promotion if generated labels are accepted without independent checks",
        },
        {
            "name": "family_leakage",
            "automatic": True,
            "criterion": "deny promotion if train/update families overlap the held-out family",
        },
        {
            "name": "missing_negative_controls",
            "automatic": True,
            "criterion": "deny promotion if no-feedback and shuffled controls are absent",
        },
    ]


def forbidden_claims() -> list[str]:
    """Return claims that `.285` wording is not allowed to make."""

    return [
        "native model-weight learning or model-weight learning from Exp 3032/3033",
        "live LLM inference, live model adaptation, or live model retraining",
        "broad autonomous self-learning beyond controller-only cached replay",
        "KAN retraining or KAN structural learning from an existing locality probe",
        "unconditional FR-11 promotion without Exp 3046 metric and rollback gates",
    ]


def inference_substrate() -> JsonDict:
    """Return the compute boundary for this governance-only aggregation run."""

    return {
        "mode": "cached_artifact_governance_aggregation",
        "cached_artifacts_only": True,
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "controller_weight_training": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "aggregation_only": True,
    }


def prior_evidence_summary(exp3032: Mapping[str, Any], exp3033: Mapping[str, Any]) -> JsonDict:
    """Summarize the evidence and the limits inherited from Exp 3032/3033."""

    return {
        "exp3032": {
            "artifact": exp3032.get("artifact", "experiment_3032_fr11_heldout_dvi_replay_v2"),
            "ready": bool(exp3032.get("fr11_heldout_replay_ready")),
            "evidence_type": "heldout_cached_exact_trace_replay",
            "heldout_trace_count": exp3032.get("heldout_trace_count", 0),
            "feasible_infeasible_auc_delta": exp3032.get("feasible_infeasible_auc_delta", 0.0),
            "shuffled_feedback_delta": exp3032.get("shuffled_feedback_delta", 0.0),
            "false_positive_delta": exp3032.get("false_positive_delta", 0.0),
            "false_negative_delta": exp3032.get("false_negative_delta", 0.0),
            "tautology_risk_cleared": bool(exp3032.get("tautology_risk_cleared")),
            "limits": _source_limits(exp3032),
        },
        "exp3033": {
            "artifact": exp3033.get(
                "artifact", "experiment_3033_fr11_nonforgetting_negative_control_stress_v1"
            ),
            "ready": bool(exp3033.get("fr11_nonforgetting_stress_ready")),
            "promotable": bool(exp3033.get("fr11_self_learning_promotable")),
            "evidence_type": "controller_only_nonforgetting_stress",
            "heldout_delta_after_update": exp3033.get("heldout_delta_after_update", 0.0),
            "prior_retention_delta": exp3033.get("prior_retention_delta", 0.0),
            "no_feedback_delta": exp3033.get("no_feedback_delta", 0.0),
            "shuffled_control_delta": exp3033.get("shuffled_control_delta", 0.0),
            "rollback_count_observed": 0,
            "kan_locality_probe_available": bool(exp3033.get("kan_locality_probe_available")),
            "limits": _source_limits(exp3033),
        },
        "limits": (
            "The prior evidence is controller-only cached replay. It supports a gated Exp 3046 "
            "protocol, but not native model-weight learning, broad continuous self-learning, "
            "or live inference claims."
        ),
    }


def _complete_artifact(
    config: ExperimentConfig,
    exp3032: Mapping[str, Any],
    exp3033: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_governance_ready": True,
        "allowed_edit_targets": allowed_edit_targets(),
        "forbidden_claims": forbidden_claims(),
        "required_metrics": required_metrics(),
        "non_promotion_criteria": non_promotion_criteria(),
        "prior_evidence_summary": prior_evidence_summary(exp3032, exp3033),
        "continuous_self_learning_scope": (
            "controller-only cached-artifact self-learning governance for .285; model weights "
            "remain out of scope until a later experiment actually trains them."
        ),
        "inference_substrate": inference_substrate(),
        "honest_verdict": "complete_fr11_governance_ready_for_exp3046",
        "field_principles": _field_principles(),
        "duration_s": duration_s,
        "tests_run": list(config.tests_run),
    }


def _blocked_artifact(
    config: ExperimentConfig,
    exp3032: Mapping[str, Any],
    exp3033: Mapping[str, Any],
    blocker: str,
    duration_s: float,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_governance_ready": False,
        "allowed_edit_targets": allowed_edit_targets(),
        "forbidden_claims": forbidden_claims(),
        "required_metrics": required_metrics(),
        "non_promotion_criteria": non_promotion_criteria(),
        "prior_evidence_summary": prior_evidence_summary(exp3032, exp3033),
        "continuous_self_learning_scope": (
            "blocked controller-only governance aggregation; model weights remain out of scope."
        ),
        "inference_substrate": inference_substrate(),
        "honest_verdict": f"blocked_{blocker}",
        "blocked_reason": blocker,
        "field_principles": _field_principles(),
        "duration_s": duration_s,
        "tests_run": list(config.tests_run),
    }


def _source_blocker(
    prefix: str,
    artifact: Mapping[str, Any],
    *,
    ready_field: str,
    promotable_field: str | None,
) -> str | None:
    if not artifact:
        return f"{prefix}_artifact_missing_or_empty"
    if artifact.get("_malformed"):
        return f"{prefix}_artifact_malformed"
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        return f"{prefix}_not_terminal"
    if artifact.get(ready_field) is not True:
        return f"{prefix}_not_ready"
    if promotable_field is not None and artifact.get(promotable_field) is not True:
        return f"{prefix}_not_controller_promotable"
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return f"{prefix}_inference_substrate_missing"
    if substrate.get("live_llm_inference") is not False:
        return f"{prefix}_live_llm_inference_claimed"
    if substrate.get("model_weight_training") is not False:
        return f"{prefix}_model_weight_training_claimed"
    return None


def _source_limits(artifact: Mapping[str, Any]) -> JsonDict:
    substrate = artifact.get("inference_substrate")
    substrate_map = substrate if isinstance(substrate, Mapping) else {}
    return {
        "controller_only": True,
        "cached_artifacts_only": bool(substrate_map.get("cached_artifacts_only", True)),
        "live_llm_inference": bool(substrate_map.get("live_llm_inference", False)),
        "model_weight_training": bool(substrate_map.get("model_weight_training", False)),
        "model_weight_mutation": bool(substrate_map.get("model_weight_training", False)),
    }


def _field_principles() -> JsonDict:
    return {
        "fr11_governance_ready": "self-learning experiments must gate on explicit promotion rules",
        "allowed_edit_targets": "controller-side and model-weight learning must be separated",
        "forbidden_claims": "capstone wording must not outrun evidence",
        "required_metrics": "downstream self-learning must be measurable",
        "non_promotion_criteria": "failure modes must be predeclared",
        "prior_evidence_summary": "governance must trace to exp3032/exp3033",
        "continuous_self_learning_scope": "FR-11 scope must be explicit",
        "inference_substrate": "aggregation work must declare no live model inference",
        "honest_verdict": "terminal verdict must be machine-readable",
    }


def _metric(name: str, measurement_rule: str) -> JsonDict:
    return {
        "name": name,
        "required_for_exp3046": True,
        "measurement_rule": measurement_rule,
    }


def _rows_by_name(rows: Any, field_name: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{field_name} must be a list")
    named: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or "name" not in row:
            raise ValueError(f"{field_name} rows must be mappings with name")
        named[str(row["name"])] = row
    return named


def _load_json(path: Path) -> JsonDict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_malformed": True}
    return payload if isinstance(payload, dict) else {"_malformed": True}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _round(value: float) -> float:
    return round(float(value), 6)
