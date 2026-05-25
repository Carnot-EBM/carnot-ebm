"""Exp 3060 solver self-model trace schema for governed FR-11 learning.

This module defines a protocol artifact, not a model update. It reads the
cached Exp 3045/3046/3047 controller-side evidence, names exactly what a future
Exp 3061 trace must record, and keeps model weights out of scope. The point is
to make StepORLM-style process feedback machine-readable without claiming that
any LLM weights learned from that feedback.

Spec refs: REQ-LEARN-3060, SCENARIO-LEARN-3060,
SCENARIO-LEARN-3060-BLOCKED.
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
ARTIFACT = "experiment_3060_fr11_solver_self_model_trace_schema_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
ARTIFACT_SCHEMA = "carnot.fr11.solver_self_model_trace_schema.v1"
TRACE_SCHEMA_ID = "carnot.fr11.solver_self_model_trace.v1"
EXP3045_ARTIFACT_REL_PATH = Path(
    "results/experiment_3045_fr11_governed_self_learning_boundary_v1.json"
)
EXP3046_ARTIFACT_REL_PATH = Path(
    "results/experiment_3046_fr11_solver_feedback_self_learning_loop_v1.json"
)
EXP3047_ARTIFACT_REL_PATH = Path("results/experiment_3047_kan_locality_nonforgetting_probe_v2.json")
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
BLOCKED_VERDICT = "blocked_missing_solver_self_model_trace_sources"
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "solver_self_model_trace_ready",
        "trace_schema",
        "allowed_edit_targets",
        "forbidden_claims",
        "validation_rules",
        "delayed_regression_window",
        "source_artifacts",
        "continuous_self_learning_scope",
        "inference_substrate",
        "honest_verdict",
    }
)
REQUIRED_TRACE_FIELDS = frozenset(
    {
        "trace_id",
        "solver_prompt_input",
        "exact_constraint_family",
        "correction_set",
        "contradiction_graph_update",
        "controller_edit",
        "rollback_decision",
        "delayed_regression_window",
        "source_artifact",
    }
)
ALLOWED_EDIT_TARGET_NAMES = frozenset(
    {
        "controller_weights",
        "trace_memory",
        "validation_thresholds",
        "contradiction_graph",
        "rollback_policy",
        "delayed_regression_manifest",
    }
)
REQUIRED_RULE_NAMES = frozenset(
    {
        "self_confirming_labels",
        "family_leakage",
        "missing_exact_authority",
        "missing_delayed_regression_evaluation",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for the schema artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3045_artifact_path: Path | None = None
    exp3046_artifact_path: Path | None = None
    exp3047_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_exp3045_artifact_path(self) -> Path:
        return self.exp3045_artifact_path or self.repo_root / EXP3045_ARTIFACT_REL_PATH

    def resolved_exp3046_artifact_path(self) -> Path:
        return self.exp3046_artifact_path or self.repo_root / EXP3046_ARTIFACT_REL_PATH

    def resolved_exp3047_artifact_path(self) -> Path:
        return self.exp3047_artifact_path or self.repo_root / EXP3047_ARTIFACT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded source artifacts for the protocol boundary."""

    exp3045_artifact: JsonDict
    exp3046_artifact: JsonDict
    exp3047_artifact: JsonDict


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3060 schema artifact."""

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
    """Load the cached controller-only evidence used to define the schema."""

    return SourceBundle(
        exp3045_artifact=_read_json(config.resolved_exp3045_artifact_path()),
        exp3046_artifact=_read_json(config.resolved_exp3046_artifact_path()),
        exp3047_artifact=_read_json(config.resolved_exp3047_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first missing or unsafe source condition before schema promotion."""

    for prefix, artifact, ready_field in (
        ("exp3045", sources.exp3045_artifact, "fr11_governance_ready"),
        ("exp3046", sources.exp3046_artifact, "fr11_solver_feedback_ready"),
        ("exp3047", sources.exp3047_artifact, "kan_locality_probe_ready"),
    ):
        blocker = _source_blocker(prefix, artifact, ready_field)
        if blocker is not None:
            return blocker
    return None


def trace_schema() -> JsonDict:
    """Return the process-level trace fields Exp 3061 can consume directly."""

    return {
        "schema_id": TRACE_SCHEMA_ID,
        "schema_version": "1.0",
        "exp3061_consumable": True,
        "for_process_level_solver_feedback": True,
        "fields": [
            _trace_field("trace_id", "string", "stable unique ID for one solver-feedback trace"),
            _trace_field(
                "solver_prompt_input",
                "object",
                "solver prompt text/ref, normalized input variables, input hash, and prompt family",
            ),
            _trace_field(
                "exact_constraint_family",
                "object",
                "family ID, held-out family ID, verifier type, and exact authority reference",
            ),
            _trace_field(
                "correction_set",
                "object",
                "solver correction set with violated constraints and suggested assignments",
            ),
            _trace_field(
                "contradiction_graph_update",
                "object",
                "nodes, edges, contradiction rate before/after, and graph hash update",
            ),
            _trace_field(
                "controller_edit",
                "object",
                "controller-side target, operation, before/after hashes, and source trace IDs",
            ),
            _trace_field(
                "rollback_decision",
                "object",
                "rollback decision, reason, comparator signal, threshold, and count delta",
            ),
            _trace_field(
                "delayed_regression_window",
                "object",
                "delayed replay IDs, minimum lag, metric name, and regression threshold",
            ),
            _trace_field(
                "source_artifact",
                "object",
                "artifact path, schema, checksum, source experiment ID, and ready field",
            ),
        ],
        "record_level_constraints": [
            "controller_edit.target MUST be one of allowed_edit_targets",
            "controller_edit.model_weight_mutation MUST be false",
            "correction_set.independent_label_authority MUST be present",
            "exact_constraint_family.exact_authority_ref MUST be present",
            "delayed_regression_window.evaluation_required MUST be true",
        ],
    }


def allowed_edit_targets() -> list[JsonDict]:
    """Return controller-side targets that a trace may record as edited."""

    return [
        _edit_target("controller_weights", "bounded inspectable controller weights"),
        _edit_target("trace_memory", "source-traced memory rows used by the controller"),
        _edit_target("validation_thresholds", "validator gates calibrated by held-out evidence"),
        _edit_target("contradiction_graph", "process graph over exact solver contradictions"),
        _edit_target("rollback_policy", "controller-side rollback thresholds and counters"),
        _edit_target(
            "delayed_regression_manifest",
            "scheduled replay windows that measure delayed failures",
        ),
    ]


def forbidden_claims() -> list[str]:
    """Return claims that Exp 3060 and Exp 3061 wording must not make."""

    return [
        "model-weight learning, model-weight mutation, or native model retraining",
        "live LLM inference, live solver inference, or live model adaptation",
        "broad autonomous self-learning beyond controller-side process traces",
        "KAN model-weight training or KAN structural learning from this schema",
        "promotion from self-confirming labels without independent exact authority",
    ]


def validation_rules() -> list[JsonDict]:
    """Return automatic failure-mode detectors required before trace promotion."""

    return [
        {
            "name": "self_confirming_labels",
            "automatic": True,
            "detects": "correction labels whose authority is the same controller edit being scored",
            "reject_when": "correction_set.independent_label_authority is missing or controller_self_label",
        },
        {
            "name": "family_leakage",
            "automatic": True,
            "detects": "train/update families overlapping held-out or delayed-regression families",
            "reject_when": "exact_constraint_family.train_family_id equals a heldout or delayed family ID",
        },
        {
            "name": "missing_exact_authority",
            "automatic": True,
            "detects": "trace rows lacking an exact SAT/SMT/solver authority reference",
            "reject_when": "exact_constraint_family.exact_authority_ref is empty",
        },
        {
            "name": "missing_delayed_regression_evaluation",
            "automatic": True,
            "detects": "updates promoted before delayed replay can measure late failures",
            "reject_when": "delayed_regression_window.evaluation_required is not true",
        },
    ]


def delayed_regression_window() -> JsonDict:
    """Return the minimum delayed-replay contract for Exp 3061 trace rows."""

    return {
        "window_id": "exp3061_minimum_delayed_replay_v1",
        "evaluation_required": True,
        "metric_name": "delayed_regression_delta",
        "failure_threshold": "delta < 0.0",
        "min_delay_cycles": 1,
        "min_replay_case_count": 1,
        "replay_case_source": "exp3046.delayed_regression",
        "source_artifact": str(EXP3046_ARTIFACT_REL_PATH),
    }


def source_artifacts(sources: SourceBundle, config: ExperimentConfig) -> list[JsonDict]:
    """Return source artifact provenance with ready fields and checksums."""

    return [
        _source_artifact_row(
            "exp3045",
            config.resolved_exp3045_artifact_path(),
            sources.exp3045_artifact,
            "fr11_governance_ready",
            "controller_governance_boundary",
            config.repo_root,
        ),
        _source_artifact_row(
            "exp3046",
            config.resolved_exp3046_artifact_path(),
            sources.exp3046_artifact,
            "fr11_solver_feedback_ready",
            "controller_solver_feedback_loop",
            config.repo_root,
        ),
        _source_artifact_row(
            "exp3047",
            config.resolved_exp3047_artifact_path(),
            sources.exp3047_artifact,
            "kan_locality_probe_ready",
            "controller_locality_nonforgetting_probe",
            config.repo_root,
        ),
    ]


def controller_only_evidence_summary(sources: SourceBundle) -> JsonDict:
    """Summarize what the sources support and what they do not support."""

    exp3045 = sources.exp3045_artifact
    exp3046 = sources.exp3046_artifact
    exp3047 = sources.exp3047_artifact
    all_ready = (
        exp3045.get("fr11_governance_ready") is True
        and exp3046.get("fr11_solver_feedback_ready") is True
        and exp3047.get("kan_locality_probe_ready") is True
    )
    return {
        "controller_only": bool(all_ready),
        "model_weight_learning_evidence": False,
        "exp3045": {
            "governance_ready": exp3045.get("fr11_governance_ready") is True,
            "allowed_controller_targets": [
                row.get("name")
                for row in _sequence(exp3045.get("allowed_edit_targets"))
                if isinstance(row, Mapping) and row.get("scope") == "allowed_controller_side"
            ],
            "forbidden_claim_count": len(_sequence(exp3045.get("forbidden_claims"))),
            "limits": _source_limits(exp3045),
        },
        "exp3046": {
            "solver_feedback_ready": exp3046.get("fr11_solver_feedback_ready") is True,
            "edit_targets_used": _sequence(exp3046.get("edit_targets_used")),
            "family_holdout_delta": exp3046.get("family_holdout_delta", 0.0),
            "delayed_regression_delta": exp3046.get("delayed_regression_delta", 0.0),
            "rollback_count": exp3046.get("rollback_count", 0),
            "flagged_adversarial": bool(exp3046.get("flagged_adversarial", False)),
            "limits": _source_limits(exp3046),
        },
        "exp3047": {
            "locality_probe_ready": exp3047.get("kan_locality_probe_ready") is True,
            "locality_metric": exp3047.get("locality_metric", 0.0),
            "changed_anchor_count": exp3047.get("changed_anchor_count", 0),
            "prior_retention_delta": exp3047.get("prior_retention_delta", 0.0),
            "limits": _source_limits(exp3047),
        },
        "interpretation": (
            "Current evidence is controller-only and cached-artifact based. "
            "Exp 3060 adds a process-level trace schema so future self-learning pilots "
            "store solver signals before claiming any stronger learning result."
        ),
    }


def inference_substrate() -> JsonDict:
    """Return the compute boundary for this cached schema-definition run."""

    return {
        "mode": "cached_artifact_schema_definition",
        "cached_artifacts_only": True,
        "schema_work_only": True,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "controller_weight_update": False,
        "trace_memory_update": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "training_scope": "none_schema_protocol_only",
    }


def schema_is_directly_consumable(
    schema: Mapping[str, Any],
    edit_targets: Sequence[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    substrate: Mapping[str, Any],
) -> bool:
    """Return whether Exp 3061 can consume this schema without extra inference."""

    field_names = {
        str(row.get("name")) for row in _sequence(schema.get("fields")) if isinstance(row, Mapping)
    }
    edit_names = {str(row.get("name")) for row in edit_targets}
    rule_names = {str(row.get("name")) for row in rules}
    return bool(
        schema.get("exp3061_consumable") is True
        and field_names == REQUIRED_TRACE_FIELDS
        and edit_names == ALLOWED_EDIT_TARGET_NAMES
        and "model_weights" not in edit_names
        and rule_names == REQUIRED_RULE_NAMES
        and window.get("evaluation_required") is True
        and int(window.get("min_delay_cycles", 0)) >= 1
        and len(sources) == 3
        and all(row.get("ready") is True for row in sources)
        and substrate.get("live_llm_inference") is False
        and substrate.get("live_model_inference") is False
        and substrate.get("model_weight_training") is False
        and substrate.get("model_weight_mutation") is False
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3060 artifact violates the trace-schema contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    ready = artifact.get("solver_self_model_trace_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if ready and not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if not ready and not verdict.startswith("blocked_"):
        raise ValueError("blocked artifacts must use a blocked_ prefix")

    schema = artifact.get("trace_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("trace_schema must be a mapping")
    field_names = {
        str(row.get("name")) for row in _sequence(schema.get("fields")) if isinstance(row, Mapping)
    }
    if field_names != REQUIRED_TRACE_FIELDS:
        raise ValueError("trace_schema must include all required trace fields")
    if schema.get("exp3061_consumable") is not True:
        raise ValueError("trace_schema must be Exp 3061 consumable")

    edit_targets = artifact.get("allowed_edit_targets")
    if not isinstance(edit_targets, list):
        raise ValueError("allowed_edit_targets must be a list")
    edit_names = {str(row.get("name")) for row in edit_targets if isinstance(row, Mapping)}
    if edit_names != ALLOWED_EDIT_TARGET_NAMES or "model_weights" in edit_names:
        raise ValueError("allowed_edit_targets must exclude model_weights")

    claims = artifact.get("forbidden_claims")
    if (
        not isinstance(claims, list)
        or not any("model-weight learning" in str(claim) for claim in claims)
        or not any("live LLM inference" in str(claim) for claim in claims)
    ):
        raise ValueError("forbidden_claims must define the claim boundary")

    rules = artifact.get("validation_rules")
    rule_names = {str(row.get("name")) for row in _sequence(rules) if isinstance(row, Mapping)}
    if rule_names != REQUIRED_RULE_NAMES or any(
        row.get("automatic") is not True for row in _sequence(rules) if isinstance(row, Mapping)
    ):
        raise ValueError("validation_rules must predeclare every required detector")

    window = artifact.get("delayed_regression_window")
    if (
        not isinstance(window, Mapping)
        or window.get("evaluation_required") is not True
        or window.get("metric_name") != "delayed_regression_delta"
        or int(window.get("min_delay_cycles", 0)) < 1
    ):
        raise ValueError("delayed_regression_window must make delayed replay measurable")

    sources = artifact.get("source_artifacts")
    if ready and (
        not isinstance(sources, list)
        or len(sources) != 3
        or any(not isinstance(row, Mapping) or row.get("ready") is not True for row in sources)
    ):
        raise ValueError("source_artifacts must show all sources ready")

    scope = str(artifact.get("continuous_self_learning_scope", ""))
    if "controller-side" not in scope or "model weights out of scope" not in scope:
        raise ValueError("continuous_self_learning_scope must state the controller-side boundary")

    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if (
        substrate.get("live_llm_inference") is not False
        or substrate.get("live_model_inference") is not False
    ):
        raise ValueError("inference_substrate must declare no live model inference")
    if (
        substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("inference_substrate must keep model weights out of scope")


def _complete_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    duration_s: float,
) -> JsonDict:
    schema = trace_schema()
    edit_targets = allowed_edit_targets()
    rules = validation_rules()
    window = delayed_regression_window()
    source_rows = source_artifacts(sources, config)
    substrate = inference_substrate()
    ready = schema_is_directly_consumable(
        schema,
        edit_targets,
        rules,
        window,
        source_rows,
        substrate,
    )
    return {
        "schema": ARTIFACT_SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "solver_self_model_trace_ready": ready,
        "trace_schema": schema,
        "allowed_edit_targets": edit_targets,
        "forbidden_claims": forbidden_claims(),
        "validation_rules": rules,
        "delayed_regression_window": window,
        "source_artifacts": source_rows,
        "controller_only_evidence_summary": controller_only_evidence_summary(sources),
        "continuous_self_learning_scope": (
            "controller-side continuous self-learning protocol/schema only; "
            "model weights out of scope until an explicit model-training experiment exists."
        ),
        "inference_substrate": substrate,
        "honest_verdict": (
            "complete_solver_self_model_trace_schema_ready"
            if ready
            else "complete_solver_self_model_trace_schema_not_ready"
        ),
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def _blocked_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    blocker: str | None,
    duration_s: float,
) -> JsonDict:
    return {
        "schema": ARTIFACT_SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "solver_self_model_trace_ready": False,
        "trace_schema": trace_schema(),
        "allowed_edit_targets": allowed_edit_targets(),
        "forbidden_claims": forbidden_claims(),
        "validation_rules": validation_rules(),
        "delayed_regression_window": delayed_regression_window(),
        "source_artifacts": [],
        "controller_only_evidence_summary": controller_only_evidence_summary(sources),
        "continuous_self_learning_scope": (
            "blocked controller-side schema aggregation; model weights out of scope."
        ),
        "inference_substrate": inference_substrate(),
        "honest_verdict": BLOCKED_VERDICT,
        "blocked_reason": blocker or "unknown_source_blocker",
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def field_principles() -> JsonDict:
    """Return compact reasons for the required terminal fields."""

    return {
        "solver_self_model_trace_ready": "self-learning pilots need a process trace schema",
        "trace_schema": "learned updates must have machine-readable provenance",
        "allowed_edit_targets": "controller-side and model-weight learning must be separated",
        "forbidden_claims": "capstone wording must not outrun evidence",
        "validation_rules": "self-learning failure modes must be predeclared",
        "delayed_regression_window": "delayed failures must be measurable",
        "source_artifacts": "schema decisions must trace to prior evidence",
        "continuous_self_learning_scope": "FR-11 scope must be explicit",
        "inference_substrate": "schema work must declare no live model inference",
        "honest_verdict": "terminal verdict must be machine-readable",
    }


def _source_blocker(prefix: str, artifact: Mapping[str, Any], ready_field: str) -> str | None:
    if not artifact:
        return f"{prefix}_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return f"{prefix}_artifact_malformed"
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        return f"{prefix}_not_terminal"
    if artifact.get(ready_field) is not True:
        return f"{prefix}_not_ready"
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return f"{prefix}_inference_substrate_missing"
    if substrate.get("live_llm_inference") is not False:
        return f"{prefix}_live_llm_inference_claimed"
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
    return {
        "experiment_id": experiment_id,
        "artifact_path": str(_relative_to(repo_root, path)),
        "artifact": artifact.get("artifact", ""),
        "schema": artifact.get("schema", ""),
        "sha256": _sha256_file(path),
        "ready_field": ready_field,
        "ready": artifact.get(ready_field) is True,
        "evidence_type": evidence_type,
        "flagged_adversarial": bool(artifact.get("flagged_adversarial", False)),
        "limits": _source_limits(artifact),
    }


def _source_limits(artifact: Mapping[str, Any]) -> JsonDict:
    substrate = artifact.get("inference_substrate")
    substrate_map = substrate if isinstance(substrate, Mapping) else {}
    return {
        "cached_artifacts_only": bool(substrate_map.get("cached_artifacts_only", True)),
        "live_llm_inference": bool(substrate_map.get("live_llm_inference", False)),
        "model_weight_training": bool(substrate_map.get("model_weight_training", False)),
        "model_weight_mutation": bool(substrate_map.get("model_weight_mutation", False)),
    }


def _trace_field(name: str, value_type: str, description: str) -> JsonDict:
    return {
        "name": name,
        "type": value_type,
        "required": True,
        "description": description,
    }


def _edit_target(name: str, reason: str) -> JsonDict:
    return {
        "name": name,
        "scope": "controller_side_only",
        "allowed": True,
        "reason": reason,
        "model_weight_mutation": False,
    }


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _round(value: float) -> float:
    return round(float(value), 6)
