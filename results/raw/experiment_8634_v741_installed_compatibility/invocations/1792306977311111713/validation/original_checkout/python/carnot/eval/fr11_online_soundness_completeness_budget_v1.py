"""Exp 3076 soundness/completeness mistake budget for FR-11 online learning.

This module defines a protocol artifact rather than a learning update. It reads
the cached FR-11 controller-side evidence from Exp 3046, Exp 3060, and Exp
3061, then turns that evidence into explicit mistake accounting for the next
tiny online pilot. The reason for keeping this as code, not prose, is that Exp
3077 needs machine-readable gates for unsafe accepts, unsafe rejects,
delayed-regression failures, controls, and rollback triggers before it can try
any controller-side update.

Spec refs: REQ-LEARN-3076, SCENARIO-LEARN-3076,
SCENARIO-LEARN-3076-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3076_fr11_online_soundness_completeness_budget_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.online_soundness_completeness_budget.v1"
EXP3046_ARTIFACT_REL_PATH = Path(
    "results/experiment_3046_fr11_solver_feedback_self_learning_loop_v1.json"
)
EXP3060_ARTIFACT_REL_PATH = Path(
    "results/experiment_3060_fr11_solver_self_model_trace_schema_v1.json"
)
EXP3061_ARTIFACT_REL_PATH = Path(
    "results/experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1.json"
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
BLOCKED_VERDICT = "blocked_missing_soundness_completeness_budget_sources"
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "soundness_completeness_budget_ready",
        "soundness_mistake_definition",
        "completeness_mistake_definition",
        "delayed_regression_window",
        "mistake_budget",
        "required_controls",
        "forbidden_claims",
        "source_artifacts",
        "continuous_self_learning_task",
        "inference_substrate",
        "honest_verdict",
    }
)
NUMERIC_BUDGET_KEYS = frozenset(
    {
        "max_soundness_mistakes",
        "max_completeness_mistakes",
        "max_delayed_regressions",
        "no_feedback_max_delta",
        "shuffled_feedback_max_delta",
        "prior_retention_floor",
        "max_contradiction_mistakes",
    }
)
REQUIRED_CONTROL_NAMES = frozenset(
    {"no_feedback_control", "shuffled_feedback_control", "prior_retention_floor"}
)
REQUIRED_ROLLBACK_TRIGGER_NAMES = frozenset(
    {
        "soundness_budget_exhausted",
        "completeness_budget_exhausted",
        "delayed_regression_budget_exhausted",
        "contradiction_budget_exhausted",
        "control_failure",
        "prior_retention_floor_breach",
    }
)
OUT_OF_SCOPE_CLAIMS = (
    "model_weight_self_learning",
    "autonomous_production_self_modification",
    "native_kan_integration",
    "native_ebt_integration",
    "live_model_inference",
    "base_model_weight_mutation",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for building the protocol artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3046_artifact_path: Path | None = None
    exp3060_artifact_path: Path | None = None
    exp3061_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_exp3046_artifact_path(self) -> Path:
        return self.exp3046_artifact_path or self.repo_root / EXP3046_ARTIFACT_REL_PATH

    def resolved_exp3060_artifact_path(self) -> Path:
        return self.exp3060_artifact_path or self.repo_root / EXP3060_ARTIFACT_REL_PATH

    def resolved_exp3061_artifact_path(self) -> Path:
        return self.exp3061_artifact_path or self.repo_root / EXP3061_ARTIFACT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded FR-11 artifacts that bound what Exp 3076 may claim."""

    exp3046_artifact: JsonDict
    exp3060_artifact: JsonDict
    exp3061_artifact: JsonDict


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3076 mistake-budget artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    duration_s = _round(active.clock() - started)
    artifact = (
        _blocked_artifact(active, sources, blocker, duration_s)
        if blocker is not None
        else _complete_artifact(active, sources, duration_s)
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load the cached controller-only sources used by the budget protocol."""

    return SourceBundle(
        exp3046_artifact=_read_json(config.resolved_exp3046_artifact_path()),
        exp3060_artifact=_read_json(config.resolved_exp3060_artifact_path()),
        exp3061_artifact=_read_json(config.resolved_exp3061_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first unsafe or missing prior artifact condition."""

    for prefix, artifact, ready_field in (
        ("exp3046", sources.exp3046_artifact, "fr11_solver_feedback_ready"),
        ("exp3060", sources.exp3060_artifact, "solver_self_model_trace_ready"),
        ("exp3061", sources.exp3061_artifact, "fr11_delayed_regression_ready"),
    ):
        blocker = _source_blocker(prefix, artifact, ready_field)
        if blocker is not None:
            return blocker
    return None


def soundness_mistake_definition() -> JsonDict:
    """Define unsafe accepts so online learning cannot count them as success."""

    return {
        "mistake_type": "soundness",
        "decision_under_audit": "controller_accept",
        "unsafe_condition": "independent_exact_authority_rejects_candidate",
        "count_key": "soundness_mistakes",
        "rate_key": "soundness_mistake_rate",
        "severity": "promotion_blocking",
        "measurement_source": "exp3077_exact_authority_replay",
        "independent_exact_authority_required": True,
        "rollback_trigger": _trigger(
            "soundness_budget_exhausted",
            "soundness_mistakes",
            "max_soundness_mistakes",
        ),
    }


def completeness_mistake_definition() -> JsonDict:
    """Define unsafe rejects/abstentions so caution is also measurable."""

    return {
        "mistake_type": "completeness",
        "decision_under_audit": "controller_reject_or_abstain",
        "unsafe_condition": "independent_exact_authority_accepts_candidate",
        "count_key": "completeness_mistakes",
        "rate_key": "completeness_mistake_rate",
        "severity": "promotion_blocking",
        "measurement_source": "exp3077_exact_authority_replay",
        "independent_exact_authority_required": True,
        "rollback_trigger": _trigger(
            "completeness_budget_exhausted",
            "completeness_mistakes",
            "max_completeness_mistakes",
        ),
    }


def delayed_regression_window() -> JsonDict:
    """Define late-failure accounting for cases replayed after an update lag."""

    return {
        "window_id": "exp3077_delayed_regression_replay_v1",
        "evaluation_required": True,
        "metric_name": "delayed_regression_mistakes",
        "source_metric": "exp3061.delayed_regression_delta",
        "min_delay_cycles": 1,
        "min_replay_case_count": 1,
        "max_allowed_mistakes": 0,
        "replay_case_source": "exp3061.split_report.delayed_regression_ids",
        "rollback_trigger": _trigger(
            "delayed_regression_budget_exhausted",
            "delayed_regressions",
            "max_delayed_regressions",
        ),
    }


def contradiction_mistake_definition() -> JsonDict:
    """Define contradiction regressions caused by a candidate controller update."""

    return {
        "mistake_type": "contradiction",
        "decision_under_audit": "candidate_controller_update",
        "unsafe_condition": "candidate_update_increases_exact_contradiction_rate",
        "count_key": "contradiction_mistakes",
        "rate_key": "contradiction_mistake_rate",
        "severity": "promotion_blocking",
        "measurement_source": "exp3077_contradiction_graph_replay",
        "rollback_trigger": _trigger(
            "contradiction_budget_exhausted",
            "contradiction_mistakes",
            "max_contradiction_mistakes",
        ),
    }


def rollback_triggers() -> list[JsonDict]:
    """Return every automatic rollback condition Exp 3077 must enforce."""

    return [
        soundness_mistake_definition()["rollback_trigger"],
        completeness_mistake_definition()["rollback_trigger"],
        delayed_regression_window()["rollback_trigger"],
        contradiction_mistake_definition()["rollback_trigger"],
        _trigger("control_failure", "control_delta", "control_max_delta"),
        _trigger(
            "prior_retention_floor_breach", "prior_retention_score", "prior_retention_floor", "<"
        ),
    ]


def mistake_budget() -> JsonDict:
    """Return numeric promotion gates for the tiny online pilot."""

    return {
        "pilot_id": "exp3077_tiny_online_controller_pilot",
        "max_soundness_mistakes": 0,
        "max_completeness_mistakes": 0,
        "max_delayed_regressions": 0,
        "max_contradiction_mistakes": 0,
        "no_feedback_max_delta": 0.0,
        "shuffled_feedback_max_delta": 0.0,
        "prior_retention_floor": 1.0,
    }


def required_controls() -> list[JsonDict]:
    """Return controls that keep feedback learning distinct from no-op gains."""

    return [
        {
            "name": "no_feedback_control",
            "required": True,
            "metric_key": "no_feedback_delta",
            "budget_key": "no_feedback_max_delta",
            "promotion_rule": "learning_delta_must_exceed_control",
        },
        {
            "name": "shuffled_feedback_control",
            "required": True,
            "metric_key": "shuffled_feedback_delta",
            "budget_key": "shuffled_feedback_max_delta",
            "promotion_rule": "learning_delta_must_exceed_control",
        },
        {
            "name": "prior_retention_floor",
            "required": True,
            "metric_key": "prior_retention_score",
            "budget_key": "prior_retention_floor",
            "promotion_rule": "retention_score_must_meet_floor",
        },
    ]


def forbidden_claims() -> list[str]:
    """Return claims that the controller-side protocol evidence cannot support."""

    return [
        "model-weight self-learning, model-weight mutation, or native model retraining",
        "autonomous production self-modification or deployment without human release gates",
        "native KAN integration, KAN model-weight training, or KAN structural learning",
        "native EBT integration, EBT model training, or live EBT inference",
        "live LLM inference, live model adaptation, or local GGUF inference in Exp 3076",
        "production self-learning promotion without exact soundness/completeness accounting",
    ]


def inference_substrate() -> JsonDict:
    """Return the execution boundary for this cached protocol-definition run."""

    return {
        "mode": "cached_artifact_protocol_definition",
        "cached_artifacts_only": True,
        "protocol_work_only": True,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "controller_weight_update": False,
        "trace_memory_update": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "training_scope": "none_protocol_definition_only",
    }


def source_artifacts(sources: SourceBundle, config: ExperimentConfig) -> list[JsonDict]:
    """Return source provenance plus controller-only/flag/out-of-scope labels."""

    return [
        _source_artifact_row(
            "exp3046",
            config.resolved_exp3046_artifact_path(),
            sources.exp3046_artifact,
            "fr11_solver_feedback_ready",
            "controller_solver_feedback_loop",
            config.repo_root,
        ),
        _source_artifact_row(
            "exp3060",
            config.resolved_exp3060_artifact_path(),
            sources.exp3060_artifact,
            "solver_self_model_trace_ready",
            "controller_trace_schema_protocol",
            config.repo_root,
        ),
        _source_artifact_row(
            "exp3061",
            config.resolved_exp3061_artifact_path(),
            sources.exp3061_artifact,
            "fr11_delayed_regression_ready",
            "controller_delayed_regression_pilot",
            config.repo_root,
        ),
    ]


def prior_artifact_claims_summary(source_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize exactly which prior claims are usable for this protocol."""

    flagged = [
        str(row.get("source_experiment_id"))
        for row in source_rows
        if _mapping(row.get("claim_classification")).get("flagged_adversarial") is True
    ]
    out_of_scope = sorted(
        {
            str(item)
            for row in source_rows
            for item in _sequence(_mapping(row.get("claim_classification")).get("out_of_scope"))
        }
    )
    all_controller_only = bool(source_rows) and all(
        _mapping(row.get("claim_classification")).get("controller_only") is True
        for row in source_rows
    )
    return {
        "all_sources_controller_only": all_controller_only,
        "flagged_source_experiments": flagged,
        "out_of_scope_claims": out_of_scope,
        "usable_claims": [
            "controller_side_protocol_evidence",
            "cached_exact_solver_feedback_accounting",
            "delayed_regression_control_requirements",
        ]
        if all_controller_only
        else [],
        "interpretation": (
            "Exp 3076 defines accounting only. Flagged prior artifacts may justify a "
            "stricter protocol, but they do not justify model-weight learning, native "
            "KAN/EBT integration, or production self-modification claims."
        ),
    }


def protocol_is_exp3077_consumable(
    soundness: Mapping[str, Any],
    completeness: Mapping[str, Any],
    delayed: Mapping[str, Any],
    budget: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    claims: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
    substrate: Mapping[str, Any],
) -> bool:
    """Return whether Exp 3077 can consume the budget without extra inference."""

    control_names = {str(row.get("name")) for row in controls}
    claim_text = " ".join(str(claim) for claim in claims)
    return bool(
        soundness.get("count_key") == "soundness_mistakes"
        and soundness.get("independent_exact_authority_required") is True
        and completeness.get("count_key") == "completeness_mistakes"
        and completeness.get("independent_exact_authority_required") is True
        and delayed.get("evaluation_required") is True
        and int(delayed.get("max_allowed_mistakes", -1)) == 0
        and NUMERIC_BUDGET_KEYS <= set(budget)
        and all(isinstance(budget[key], int | float) for key in NUMERIC_BUDGET_KEYS)
        and control_names == REQUIRED_CONTROL_NAMES
        and "model-weight self-learning" in claim_text
        and "native KAN integration" in claim_text
        and "native EBT integration" in claim_text
        and len(sources) == 3
        and all(row.get("ready") is True for row in sources)
        and all(
            _mapping(row.get("claim_classification")).get("controller_only") is True
            for row in sources
        )
        and substrate.get("live_llm_inference") is False
        and substrate.get("live_model_inference") is False
        and substrate.get("model_weight_training") is False
        and substrate.get("model_weight_mutation") is False
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3076 artifact violates the budget protocol contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_task") is not True:
        raise ValueError("continuous_self_learning_task must be true")

    ready = artifact.get("soundness_completeness_budget_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if ready and not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if not ready and verdict != BLOCKED_VERDICT:
        raise ValueError("blocked artifacts must use the blocked budget verdict")

    soundness = artifact.get("soundness_mistake_definition")
    if (
        not isinstance(soundness, Mapping)
        or soundness.get("count_key") != "soundness_mistakes"
        or soundness.get("independent_exact_authority_required") is not True
    ):
        raise ValueError("soundness_mistake_definition is incomplete")

    completeness = artifact.get("completeness_mistake_definition")
    if (
        not isinstance(completeness, Mapping)
        or completeness.get("count_key") != "completeness_mistakes"
        or completeness.get("independent_exact_authority_required") is not True
    ):
        raise ValueError("completeness_mistake_definition is incomplete")

    delayed = artifact.get("delayed_regression_window")
    if (
        not isinstance(delayed, Mapping)
        or delayed.get("evaluation_required") is not True
        or delayed.get("metric_name") != "delayed_regression_mistakes"
        or int(delayed.get("max_allowed_mistakes", -1)) != 0
    ):
        raise ValueError("delayed_regression_window must define zero late failures")

    budget = artifact.get("mistake_budget")
    if (
        not isinstance(budget, Mapping)
        or not NUMERIC_BUDGET_KEYS <= set(budget)
        or any(not isinstance(budget[key], int | float) for key in NUMERIC_BUDGET_KEYS)
    ):
        raise ValueError("mistake_budget must contain numeric promotion gates")

    controls = artifact.get("required_controls")
    control_names = {
        str(row.get("name")) for row in _sequence(controls) if isinstance(row, Mapping)
    }
    if control_names != REQUIRED_CONTROL_NAMES:
        raise ValueError("required_controls must include no-feedback, shuffled, and prior gates")

    claims = artifact.get("forbidden_claims")
    claim_text = " ".join(str(claim) for claim in _sequence(claims))
    if (
        "model-weight self-learning" not in claim_text
        or "native KAN integration" not in claim_text
        or "native EBT integration" not in claim_text
    ):
        raise ValueError("forbidden_claims must name unsupported stronger claims")

    sources = artifact.get("source_artifacts")
    if not isinstance(sources, list):
        raise ValueError("source_artifacts must be a list")
    if ready and (
        len(sources) != 3
        or any(
            not isinstance(row, Mapping)
            or row.get("ready") is not True
            or _mapping(row.get("claim_classification")).get("controller_only") is not True
            for row in sources
        )
    ):
        raise ValueError("source_artifacts must be ready controller-only sources")

    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if (
        substrate.get("live_llm_inference") is not False
        or substrate.get("live_model_inference") is True
        or substrate.get("model_weight_training") is True
        or substrate.get("model_weight_mutation") is True
    ):
        raise ValueError("inference_substrate must exclude live/model-weight updates")

    if ready and not protocol_is_exp3077_consumable(
        soundness,
        completeness,
        delayed,
        budget,
        _sequence(controls),
        _sequence(claims),
        sources,
        substrate,
    ):
        raise ValueError("soundness_completeness_budget_ready requires Exp 3077 consumability")


def _complete_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    duration_s: float,
) -> JsonDict:
    source_rows = source_artifacts(sources, config)
    substrate = inference_substrate()
    ready = protocol_is_exp3077_consumable(
        soundness_mistake_definition(),
        completeness_mistake_definition(),
        delayed_regression_window(),
        mistake_budget(),
        required_controls(),
        forbidden_claims(),
        source_rows,
        substrate,
    )
    return _base_artifact(
        config,
        source_rows,
        ready=ready,
        duration_s=duration_s,
        substrate=substrate,
        verdict=(
            "complete_fr11_soundness_completeness_budget_ready"
            if ready
            else "complete_fr11_soundness_completeness_budget_not_ready"
        ),
    )


def _blocked_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    blocker: str | None,
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        config,
        source_artifacts(sources, config),
        ready=False,
        duration_s=duration_s,
        substrate=inference_substrate(),
        verdict=BLOCKED_VERDICT,
    )
    artifact["blocked_reason"] = blocker or "unknown_source_blocker"
    return artifact


def _base_artifact(
    config: ExperimentConfig,
    source_rows: list[JsonDict],
    *,
    ready: bool,
    duration_s: float,
    substrate: Mapping[str, Any],
    verdict: str,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "soundness_completeness_budget_ready": ready,
        "soundness_mistake_definition": soundness_mistake_definition(),
        "completeness_mistake_definition": completeness_mistake_definition(),
        "delayed_regression_window": delayed_regression_window(),
        "contradiction_mistake_definition": contradiction_mistake_definition(),
        "rollback_triggers": rollback_triggers(),
        "mistake_budget": mistake_budget(),
        "required_controls": required_controls(),
        "forbidden_claims": forbidden_claims(),
        "source_artifacts": source_rows,
        "prior_artifact_claims_summary": prior_artifact_claims_summary(source_rows),
        "continuous_self_learning_task": True,
        "inference_substrate": dict(substrate),
        "honest_verdict": verdict,
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def field_principles() -> JsonDict:
    """Return compact reasons for the required terminal fields."""

    return {
        "soundness_completeness_budget_ready": (
            "FR-11 online learning needs explicit safety accounting"
        ),
        "soundness_mistake_definition": "unsafe accepts must be defined",
        "completeness_mistake_definition": "unsafe rejects/abstentions must be defined",
        "delayed_regression_window": "delayed failures must be measurable",
        "mistake_budget": "promotion gates must be numeric",
        "required_controls": "learning must beat no-feedback and shuffled-feedback controls",
        "forbidden_claims": "controller learning must not imply model-weight learning",
        "source_artifacts": "protocol must trace to prior evidence",
        "continuous_self_learning_task": "milestone self-learning requirement must be explicit",
        "inference_substrate": "protocol work must declare no live model inference",
        "honest_verdict": "terminal verdict must start with a success prefix unless blocked",
    }


def _source_blocker(prefix: str, artifact: Mapping[str, Any], ready_field: str) -> str | None:
    if not artifact:
        return f"{prefix}_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return f"{prefix}_artifact_malformed"
    if not _is_terminal(artifact):
        return f"{prefix}_not_terminal"
    if artifact.get(ready_field) is not True:
        return f"{prefix}_not_ready"
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return f"{prefix}_inference_substrate_missing"
    if substrate.get("live_llm_inference") is True or substrate.get("live_model_inference") is True:
        return f"{prefix}_live_model_inference_claimed"
    if substrate.get("model_weight_training") is True:
        return f"{prefix}_model_weight_training_claimed"
    if substrate.get("model_weight_mutation") is True:
        return f"{prefix}_model_weight_mutation_claimed"
    return None


def _source_artifact_row(
    experiment_id: str,
    path: Path,
    artifact: Mapping[str, Any],
    ready_field: str,
    evidence_type: str,
    repo_root: Path,
) -> JsonDict:
    ready = artifact.get(ready_field) is True
    return {
        "source_experiment_id": experiment_id,
        "artifact_path": str(_relative_to(repo_root, path)),
        "artifact": str(artifact.get("artifact", "")),
        "schema": str(artifact.get("schema", "")),
        "sha256": _file_sha256(path),
        "ready_field": ready_field,
        "ready": ready,
        "evidence_type": evidence_type,
        "claim_classification": _claim_classification(artifact, ready),
    }


def _claim_classification(artifact: Mapping[str, Any], ready: bool) -> JsonDict:
    flagged = bool(artifact.get("flagged_adversarial", False))
    controller_only = bool(
        ready and not _source_model_weight_claimed(artifact) and not _live_claimed(artifact)
    )
    status = "blocked_or_missing"
    if controller_only:
        status = "controller_only_flagged" if flagged else "controller_only_unflagged"
    return {
        "controller_only": controller_only,
        "flagged_adversarial": flagged,
        "status": status,
        "out_of_scope": list(OUT_OF_SCOPE_CLAIMS),
    }


def _trigger(
    name: str,
    trigger_key: str,
    threshold_key: str,
    comparison: str = ">",
) -> JsonDict:
    return {
        "name": name,
        "trigger_key": trigger_key,
        "threshold_key": threshold_key,
        "comparison": comparison,
        "action": "rollback_candidate_update",
    }


def _is_terminal(artifact: Mapping[str, Any]) -> bool:
    return str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES)


def _source_model_weight_claimed(artifact: Mapping[str, Any]) -> bool:
    substrate = _mapping(artifact.get("inference_substrate"))
    return bool(
        artifact.get("model_weight_training") is True
        or artifact.get("model_weight_mutation") is True
        or substrate.get("model_weight_training") is True
        or substrate.get("model_weight_mutation") is True
    )


def _live_claimed(artifact: Mapping[str, Any]) -> bool:
    substrate = _mapping(artifact.get("inference_substrate"))
    return bool(
        artifact.get("live_llm_inference") is True
        or artifact.get("live_model_inference") is True
        or substrate.get("live_llm_inference") is True
        or substrate.get("live_model_inference") is True
    )


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {"_malformed": True}
    return dict(payload) if isinstance(payload, Mapping) else {"_malformed": True}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _round(value: float) -> float:
    return round(float(value), 6)
