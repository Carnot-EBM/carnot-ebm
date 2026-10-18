"""Exp 3047 KAN-style locality/nonforgetting probe over Exp 3046.

This module is deliberately bounded. It does not train a KAN, mutate model
weights, or run live LLM inference. It replays the governed Exp 3046
controller-side solver-feedback update and treats inspectable controller weight
keys as the smallest available anchor structure for a KAN-style locality probe.

Spec refs: REQ-LEARN-3047, SCENARIO-LEARN-3047,
SCENARIO-LEARN-3047-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from carnot.eval import fr11_solver_feedback_self_learning_loop_v1 as exp3046


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3047_kan_locality_nonforgetting_probe_v2"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.kan_locality_nonforgetting_probe.v2"
EXP3046_ARTIFACT_REL_PATH = Path(
    "results/experiment_3046_fr11_solver_feedback_self_learning_loop_v1.json"
)
EXP3044_ARTIFACT_REL_PATH = exp3046.EXP3044_ARTIFACT_REL_PATH
EXP3045_ARTIFACT_REL_PATH = exp3046.EXP3045_ARTIFACT_REL_PATH
LOCALITY_REPORT_REL_PATH = Path(
    "results/kan_locality_nonforgetting_probe_3047/locality_report.jsonl"
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
BLOCKED_VERDICT = "blocked_missing_solver_feedback_locality_source"
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "kan_locality_probe_ready",
        "locality_metric",
        "changed_anchor_count",
        "anchored_prior_count",
        "heldout_delta",
        "prior_retention_delta",
        "irrelevant_control_delta",
        "comparator_summary",
        "promotion_decision",
        "inference_substrate",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for Exp 3047."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    locality_report_path: Path | None = None
    exp3046_artifact_path: Path | None = None
    exp3044_artifact_path: Path | None = None
    exp3045_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_locality_report_path(self) -> Path:
        return self.locality_report_path or self.repo_root / LOCALITY_REPORT_REL_PATH

    def resolved_exp3046_artifact_path(self) -> Path:
        return self.exp3046_artifact_path or self.repo_root / EXP3046_ARTIFACT_REL_PATH

    def resolved_exp3044_artifact_path(self) -> Path:
        return self.exp3044_artifact_path or self.repo_root / EXP3044_ARTIFACT_REL_PATH

    def resolved_exp3045_artifact_path(self) -> Path:
        return self.exp3045_artifact_path or self.repo_root / EXP3045_ARTIFACT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded source artifacts required to replay the Exp 3046 controller update."""

    exp3046_artifact: JsonDict
    exp3044_artifact: JsonDict
    exp3045_artifact: JsonDict


@dataclass(frozen=True)
class LocalityProbe:
    """Measured locality and retention fields for the governed controller update."""

    locality_metric: float
    changed_anchor_count: int
    anchored_prior_count: int
    heldout_delta: float
    prior_retention_delta: float
    irrelevant_control_delta: float
    changed_anchor_keys: tuple[str, ...]
    anchored_prior_keys: tuple[str, ...]
    total_anchor_count: int
    changed_update_trace_count: int
    comparator_summary: JsonDict
    split_report: JsonDict
    source_trace_counts: JsonDict
    split_reused_from_exp3046: bool


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3047 locality probe artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        artifact = blocked_artifact(active, blocker, _round(active.clock() - started))
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    probe = measure_locality(sources)
    artifact = complete_artifact(active, sources, probe, _round(active.clock() - started))
    validate_artifact(artifact)
    _write_jsonl(active.resolved_locality_report_path(), _locality_report_rows(artifact))
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load Exp 3046 plus the source artifacts needed to rebuild its split."""

    return SourceBundle(
        exp3046_artifact=_read_json(config.resolved_exp3046_artifact_path()),
        exp3044_artifact=_read_json(config.resolved_exp3044_artifact_path()),
        exp3045_artifact=_read_json(config.resolved_exp3045_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first missing or unsafe source condition before replay."""

    exp3046_blocker = _exp3046_blocker(sources.exp3046_artifact)
    if exp3046_blocker is not None:
        return exp3046_blocker
    exp3044_blocker = _exp3044_blocker(sources.exp3044_artifact)
    if exp3044_blocker is not None:
        return exp3044_blocker
    return _exp3045_blocker(sources.exp3045_artifact)


def measure_locality(sources: SourceBundle) -> LocalityProbe:
    """Replay Exp 3046 and measure changed versus anchored controller weights."""

    blocker = precondition_blocker(sources)
    if blocker is not None:
        raise ValueError(f"cannot measure locality with blocked sources: {blocker}")

    split = exp3046.build_family_split(sources.exp3044_artifact)
    baseline = exp3046.initial_controller_state()
    result = exp3046.run_governed_loop(
        split,
        sources.exp3044_artifact,
        sources.exp3045_artifact,
    )
    updated = result.updated_state
    all_anchor_keys = tuple(sorted(set(baseline.weights) | set(updated.weights)))
    changed_anchor_keys = tuple(
        key
        for key in all_anchor_keys
        if float(baseline.weights.get(key, 0.0)) != float(updated.weights.get(key, 0.0))
    )
    anchored_prior_keys = tuple(
        sorted(
            {
                feature
                for case in split.prior_exact
                for feature in case.features
                if float(baseline.weights.get(feature, 0.0))
                == float(updated.weights.get(feature, 0.0))
                and exp3046.predicted_valid(updated.weights, case) is case.expected_valid
            }
        )
    )
    total_anchor_count = len(all_anchor_keys)
    locality_metric = _round(1.0 - (len(changed_anchor_keys) / total_anchor_count))
    exp3046_split = _mapping(sources.exp3046_artifact.get("split_report"))
    reused_train_ids = [case.case_id for case in split.train_update]
    reused_holdout_ids = [case.case_id for case in split.family_holdout]
    matches_exp3046 = reused_train_ids == _sequence(
        exp3046_split.get("train_update_ids")
    ) and reused_holdout_ids == _sequence(exp3046_split.get("family_holdout_ids"))
    source_trace_counts = {
        "exp3046_train_update_case_count": len(split.train_update),
        "exp3046_family_holdout_case_count": len(split.family_holdout),
        "exp3046_prior_exact_case_count": len(split.prior_exact),
        "exp3046_source_traced_update_count": len(updated.trace_memory),
    }
    return LocalityProbe(
        locality_metric=locality_metric,
        changed_anchor_count=len(changed_anchor_keys),
        anchored_prior_count=len(anchored_prior_keys),
        heldout_delta=float(result.metrics["family_holdout_delta"]),
        prior_retention_delta=float(result.metrics["prior_retention_delta"]),
        irrelevant_control_delta=float(result.metrics["delayed_regression_delta"]),
        changed_anchor_keys=changed_anchor_keys,
        anchored_prior_keys=anchored_prior_keys,
        total_anchor_count=total_anchor_count,
        changed_update_trace_count=len(updated.trace_memory),
        comparator_summary=_comparator_summary(split, baseline),
        split_report={
            "reused_train_update_ids": reused_train_ids,
            "reused_family_holdout_ids": reused_holdout_ids,
            "matches_exp3046_artifact": matches_exp3046,
        },
        source_trace_counts=source_trace_counts,
        split_reused_from_exp3046=matches_exp3046,
    )


def complete_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    probe: LocalityProbe,
    duration_s: float,
) -> JsonDict:
    """Build the complete Exp 3047 artifact from a measured locality probe."""

    ready = bool(
        probe.locality_metric > 0.0
        and probe.changed_anchor_count > 0
        and probe.anchored_prior_count > 0
        and probe.heldout_delta > 0.0
        and probe.prior_retention_delta >= 0.0
        and probe.irrelevant_control_delta == 0.0
        and _mapping(probe.comparator_summary).get("available") is True
        and probe.split_reused_from_exp3046
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "kan_locality_probe_ready": ready,
        "locality_metric": probe.locality_metric,
        "changed_anchor_count": probe.changed_anchor_count,
        "anchored_prior_count": probe.anchored_prior_count,
        "heldout_delta": probe.heldout_delta,
        "prior_retention_delta": probe.prior_retention_delta,
        "irrelevant_control_delta": probe.irrelevant_control_delta,
        "comparator_summary": probe.comparator_summary,
        "promotion_decision": (
            "controller_locality_evidence_only" if ready else "controller_locality_not_promoted"
        ),
        "inference_substrate": inference_substrate(controller_weight_update=True),
        "honest_verdict": (
            "complete_kan_locality_controller_probe_ready"
            if ready
            else "complete_kan_locality_controller_probe_not_promoted"
        ),
        "locality_report": {
            "total_anchor_count": probe.total_anchor_count,
            "changed_anchor_keys": list(probe.changed_anchor_keys),
            "anchored_prior_keys": list(probe.anchored_prior_keys),
            "changed_update_trace_count": probe.changed_update_trace_count,
            "locality_metric_definition": "1 - changed_controller_anchor_count / total_anchor_count",
        },
        "split_report": probe.split_report,
        "source_trace_counts": probe.source_trace_counts,
        "source_artifacts": {
            "exp3046_artifact": str(
                _relative_to(config.repo_root, config.resolved_exp3046_artifact_path())
            ),
            "exp3046_ready": sources.exp3046_artifact.get("fr11_solver_feedback_ready") is True,
            "exp3044_artifact": str(
                _relative_to(config.repo_root, config.resolved_exp3044_artifact_path())
            ),
            "exp3044_ready": sources.exp3044_artifact.get("validator_tree_exactness_ready") is True,
            "exp3045_artifact": str(
                _relative_to(config.repo_root, config.resolved_exp3045_artifact_path())
            ),
            "exp3045_ready": sources.exp3045_artifact.get("fr11_governance_ready") is True,
        },
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def blocked_artifact(config: ExperimentConfig, reason: str, duration_s: float) -> JsonDict:
    """Build a fail-closed artifact when source evidence is unavailable."""

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "kan_locality_probe_ready": False,
        "locality_metric": 0.0,
        "changed_anchor_count": 0,
        "anchored_prior_count": 0,
        "heldout_delta": 0.0,
        "prior_retention_delta": 0.0,
        "irrelevant_control_delta": 0.0,
        "comparator_summary": {
            "available": False,
            "reason": "blocked_before_comparator",
            "blocked_reason": reason,
        },
        "promotion_decision": "blocked",
        "inference_substrate": inference_substrate(controller_weight_update=False),
        "honest_verdict": BLOCKED_VERDICT,
        "blocked_reason": reason,
        "locality_report": {
            "total_anchor_count": 0,
            "changed_anchor_keys": [],
            "anchored_prior_keys": [],
            "changed_update_trace_count": 0,
            "locality_metric_definition": "not_measured_blocked_source",
        },
        "source_trace_counts": {
            "exp3046_train_update_case_count": 0,
            "exp3046_family_holdout_case_count": 0,
            "exp3046_prior_exact_case_count": 0,
            "exp3046_source_traced_update_count": 0,
        },
        "field_principles": field_principles(),
        "tests_run": list(config.tests_run),
        "duration_s": duration_s,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3047 artifact violates the locality-probe contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if substrate.get("live_llm_inference") is not False:
        raise ValueError("live LLM inference must remain false")
    if (
        substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("model weights must not be trained or mutated")

    ready = artifact.get("kan_locality_probe_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if not ready:
        if verdict != BLOCKED_VERDICT:
            raise ValueError("blocked artifacts cannot be ready without the blocked verdict")
        return
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if float(artifact.get("locality_metric") or 0.0) <= 0.0:
        raise ValueError("locality_metric must be positive for ready artifacts")
    if int(artifact.get("changed_anchor_count") or 0) <= 0:
        raise ValueError("changed_anchor_count must be positive")
    if int(artifact.get("anchored_prior_count") or 0) <= 0:
        raise ValueError("anchored_prior_count must be positive")
    if float(artifact.get("heldout_delta") or 0.0) <= 0.0:
        raise ValueError("heldout_delta must be positive")
    if float(artifact.get("prior_retention_delta") or 0.0) < 0.0:
        raise ValueError("prior_retention_delta must not regress")
    if float(artifact.get("irrelevant_control_delta") or 0.0) != 0.0:
        raise ValueError("irrelevant_control_delta must remain stable")
    comparator = artifact.get("comparator_summary")
    if not isinstance(comparator, Mapping) or comparator.get("available") is not True:
        raise ValueError("comparator_summary must contain a measured comparator")
    if comparator.get("promoted") is True:
        raise ValueError("comparator_summary must not promote control updates")


def inference_substrate(*, controller_weight_update: bool) -> JsonDict:
    """Return the execution boundary so the artifact cannot imply LLM retraining."""

    return {
        "mode": "cached_exp3046_controller_locality_probe",
        "cached_artifacts_only": True,
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "controller_weight_update": controller_weight_update,
        "trace_memory_update": controller_weight_update,
        "kan_model_weight_training": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "training_scope": "bounded_controller_side_locality_probe",
    }


def field_principles() -> JsonDict:
    """Return compact reasons for the required terminal fields."""

    return {
        "kan_locality_probe_ready": "matrix v19 needs a bounded locality result",
        "locality_metric": "KAN/locality claims need a numeric locality measure",
        "changed_anchor_count": "locality must identify how much changed",
        "anchored_prior_count": "nonforgetting anchors must be counted",
        "heldout_delta": "locality must preserve or improve related cases",
        "prior_retention_delta": "locality must not erase prior cases",
        "irrelevant_control_delta": "locality must not drift unrelated cases",
        "comparator_summary": "missing comparators must be explicit",
        "promotion_decision": "controller/locality evidence must not imply model-weight learning",
        "inference_substrate": "controller probe must not be confused with live LLM inference",
        "honest_verdict": "terminal verdict must be machine-readable",
    }


def _comparator_summary(split: exp3046.FamilySplit, baseline: exp3046.ControllerState) -> JsonDict:
    shuffled_update = exp3046.apply_feedback_updates(
        baseline,
        _shuffled_feedback_update_cases(split.train_update),
    )
    baseline_holdout = exp3046.mean_signed_margin(baseline.weights, split.family_holdout)
    shuffled_holdout = exp3046.mean_signed_margin(shuffled_update.weights, split.family_holdout)
    changed = _changed_anchor_keys(baseline.weights, shuffled_update.weights)
    total_anchor_count = len(set(baseline.weights) | set(shuffled_update.weights))
    return {
        "available": True,
        "comparator_type": "shuffled_control_update",
        "reason": "measured_exp3046_shuffled_feedback_control",
        "heldout_delta": _round(shuffled_holdout - baseline_holdout),
        "changed_anchor_count": len(changed),
        "locality_metric": _round(1.0 - (len(changed) / total_anchor_count)),
        "promoted": shuffled_holdout > baseline_holdout,
    }


def _changed_anchor_keys(
    baseline_weights: Mapping[str, float],
    updated_weights: Mapping[str, float],
) -> tuple[str, ...]:
    keys = sorted(set(baseline_weights) | set(updated_weights))
    return tuple(
        key
        for key in keys
        if float(baseline_weights.get(key, 0.0)) != float(updated_weights.get(key, 0.0))
    )


def _shuffled_feedback_update_cases(
    cases: Sequence[exp3046.SatCase],
) -> tuple[exp3046.SatCase, ...]:
    labels = [case.expected_valid for case in cases]
    shifted = labels[1:] + labels[:1]
    return tuple(
        exp3046.SatCase(
            case_id=f"shuffled-{case.case_id}",
            split="shuffled_feedback_control",
            family=case.family,
            a=case.a,
            b=case.b,
            total=case.total,
            expected_valid=label,
            source_trace_id=f"control:shuffled:{case.case_id}",
            features=case.features,
        )
        for case, label in zip(cases, shifted, strict=True)
    )


def _exp3046_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3046_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3046_artifact_malformed"
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        return "exp3046_not_terminal"
    if artifact.get("fr11_solver_feedback_ready") is not True:
        return "exp3046_solver_feedback_not_ready"
    counts = artifact.get("source_trace_counts")
    if not isinstance(counts, Mapping) or int(counts.get("source_traced_update_count", 0)) <= 0:
        return "exp3046_source_trace_counts_missing"
    return _substrate_blocker("exp3046", artifact)


def _exp3044_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3044_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3044_artifact_malformed"
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        return "exp3044_not_terminal"
    if artifact.get("validator_tree_exactness_ready") is not True:
        return "exp3044_exact_feedback_not_ready"
    if not _sequence(artifact.get("correction_sets")):
        return "exp3044_correction_sets_missing"
    return _substrate_blocker("exp3044", artifact)


def _exp3045_blocker(artifact: Mapping[str, Any]) -> str | None:
    if not artifact:
        return "exp3045_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return "exp3045_artifact_malformed"
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        return "exp3045_not_terminal"
    if artifact.get("fr11_governance_ready") is not True:
        return "exp3045_governance_not_ready"
    return _substrate_blocker("exp3045", artifact)


def _substrate_blocker(prefix: str, artifact: Mapping[str, Any]) -> str | None:
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


def _locality_report_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "section": "locality",
            "locality_metric": artifact["locality_metric"],
            "changed_anchor_count": artifact["changed_anchor_count"],
        },
        {
            "section": "retention",
            "anchored_prior_count": artifact["anchored_prior_count"],
            "prior_retention_delta": artifact["prior_retention_delta"],
        },
        {
            "section": "comparator",
            "comparator_summary": artifact["comparator_summary"],
        },
    ]


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _round(value: float) -> float:
    return round(float(value), 6)


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path
