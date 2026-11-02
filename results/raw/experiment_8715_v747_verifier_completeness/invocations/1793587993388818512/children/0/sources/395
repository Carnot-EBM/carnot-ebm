"""Exp 3033 FR-11 nonforgetting and negative-control stress.

This module stays inside the cached-feedback boundary.  It takes the bounded
Exp 3020 controller weights, applies the Exp 3032 held-out feedback as another
inspectable controller update, and asks whether old exact traces still score
correctly.  The result is evidence about the controller update path only: no
LLM is queried, no model weights are trained, and KAN locality is reported only
when an existing local KAN probe can actually be run.

Spec refs: REQ-LEARN-3033, SCENARIO-LEARN-3033,
SCENARIO-LEARN-3033-BOUNDED, SCENARIO-LEARN-3033-BLOCKED.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3033_fr11_nonforgetting_negative_control_stress_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.nonforgetting_negative_control_stress.v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME

EXP3032_ARTIFACT_REL_PATH = Path("results/experiment_3032_fr11_heldout_dvi_replay_v2.json")
HELDOUT_REPLAY_REL_PATH = Path("results/fr11_heldout_dvi_replay_3032/heldout_replay.jsonl")
EXP3020_ARTIFACT_REL_PATH = Path(
    "results/experiment_3020_dvi_verifier_feedback_self_learning_controller_v1.json"
)
CONTROLLER_CONFIG_REL_PATH = Path(
    "results/dvi_verifier_feedback_self_learning_controller_3020/controller_config.json"
)
REPLAY_TRANSCRIPT_REL_PATH = Path(
    "results/dvi_verifier_feedback_self_learning_controller_3020/replay_transcript.jsonl"
)
STRESS_REPORT_REL_PATH = Path(
    "results/fr11_nonforgetting_negative_control_stress_3033/stress_report.jsonl"
)

LEARNING_RATE = 0.25
MAX_ABS_WEIGHT = 1.0
RETENTION_TOLERANCE = 0.05
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
ALLOWED_FEATURE_PREFIXES = (
    "authority::",
    "category::",
    "evidence::",
    "failing_node_kind::",
    "failure_reason::",
    "frontier::",
    "memory_guard::",
    "node_kind::",
)
PROHIBITED_FEATURE_NAMES = frozenset(
    {
        "candidate_role",
        "certificate_status",
        "heldout_success_label",
        "item_id",
        "row_id",
        "original_controller_rationale",
        "update_accepted",
    }
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_nonforgetting_stress_ready",
        "fr11_self_learning_promotable",
        "prior_retention_delta",
        "heldout_delta_after_update",
        "shuffled_control_delta",
        "no_feedback_delta",
        "drift_failures",
        "kan_locality_probe_available",
        "inference_substrate",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for Exp 3033.

    The config is path-heavy because the experiment audits persisted artifacts,
    not in-memory fixtures.  Tests can point every source to a temporary
    workspace while the production run uses the checked-in results directory.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    stress_report_path: Path | None = None
    exp3032_artifact_path: Path | None = None
    heldout_replay_path: Path | None = None
    exp3020_artifact_path: Path | None = None
    controller_config_path: Path | None = None
    replay_transcript_path: Path | None = None
    retention_tolerance: float = RETENTION_TOLERANCE
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_stress_report_path(self) -> Path:
        return self.stress_report_path or self.repo_root / STRESS_REPORT_REL_PATH

    def resolved_exp3032_artifact_path(self) -> Path:
        return self.exp3032_artifact_path or self.repo_root / EXP3032_ARTIFACT_REL_PATH

    def resolved_heldout_replay_path(self) -> Path:
        return self.heldout_replay_path or self.repo_root / HELDOUT_REPLAY_REL_PATH

    def resolved_exp3020_artifact_path(self) -> Path:
        return self.exp3020_artifact_path or self.repo_root / EXP3020_ARTIFACT_REL_PATH

    def resolved_controller_config_path(self) -> Path:
        return self.controller_config_path or self.repo_root / CONTROLLER_CONFIG_REL_PATH

    def resolved_replay_transcript_path(self) -> Path:
        return self.replay_transcript_path or self.repo_root / REPLAY_TRANSCRIPT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded Exp 3032/3020 evidence used by the stress run."""

    exp3032_artifact: JsonDict
    heldout_replay_rows: tuple[JsonDict, ...]
    exp3020_artifact: JsonDict
    controller_config: JsonDict
    replay_transcript_rows: tuple[JsonDict, ...]


@dataclass(frozen=True)
class FeedbackTrace:
    """A label-bearing exact feedback trace with only controller-safe features."""

    trace_id: str
    source_experiment: str
    item_id: str
    row_id: str
    expected_feedback: bool
    exact_machine_checked: bool
    features: tuple[str, ...]


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3033 terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        artifact = _blocked_artifact(active, _round(active.clock() - started), blocker)
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    prior_traces = build_prior_exact_traces(sources)
    heldout_traces = build_heldout_feedback_traces(sources)
    trace_blocker = trace_precondition_blocker(prior_traces, heldout_traces)
    if trace_blocker is not None:
        artifact = _blocked_artifact(active, _round(active.clock() - started), trace_blocker)
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    stress = stress_controller(
        prior_traces,
        heldout_traces,
        sources.controller_config,
        retention_tolerance=active.retention_tolerance,
    )
    locality = kan_locality_probe()
    promotable = bool(
        stress["heldout_delta_after_update"] > 0.0
        and stress["shuffled_control_delta"] <= 0.0
        and stress["adversarial_irrelevant_control_delta"] <= 0.0
        and stress["no_feedback_delta"] == 0.0
        and stress["prior_retention_delta"] >= -active.retention_tolerance
        and not stress["drift_failures"]
    )
    _write_jsonl(active.resolved_stress_report_path(), _stress_report_rows(stress))
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_nonforgetting_stress_ready": True,
        "fr11_self_learning_promotable": promotable,
        "prior_retention_delta": stress["prior_retention_delta"],
        "heldout_delta_after_update": stress["heldout_delta_after_update"],
        "shuffled_control_delta": stress["shuffled_control_delta"],
        "adversarial_irrelevant_control_delta": stress["adversarial_irrelevant_control_delta"],
        "no_feedback_delta": stress["no_feedback_delta"],
        "drift_failures": stress["drift_failures"],
        "kan_locality_probe_available": bool(locality["available"]),
        "kan_locality_report": locality,
        "inference_substrate": inference_substrate(controller_weight_update=True),
        "honest_verdict": (
            "complete_controller_only_promotable"
            if promotable
            else "complete_controller_only_not_promoted"
        ),
        "promotion_decision": (
            "controller_only_promotable" if promotable else "controller_only_bounded"
        ),
        "retention_tolerance": active.retention_tolerance,
        "stress_report_path": str(
            _relative_to(active.repo_root, active.resolved_stress_report_path())
        ),
        "source_trace_counts": {
            "prior_exact_trace_count": len(prior_traces),
            "heldout_trace_count": len(heldout_traces),
            "controller_weight_count": len(_controller_weights(sources.controller_config)),
        },
        "prior_retention_report": stress["prior_retention_report"],
        "heldout_update_report": stress["heldout_update_report"],
        "control_report": stress["control_report"],
        "source_artifacts": {
            "exp3032_artifact": str(
                _relative_to(active.repo_root, active.resolved_exp3032_artifact_path())
            ),
            "exp3032_ready": sources.exp3032_artifact.get("fr11_heldout_replay_ready") is True,
            "exp3020_artifact": str(
                _relative_to(active.repo_root, active.resolved_exp3020_artifact_path())
            ),
            "exp3020_ready": sources.exp3020_artifact.get("verifier_feedback_controller_ready")
            is True,
            "controller_config_path": str(
                _relative_to(active.repo_root, active.resolved_controller_config_path())
            ),
            "replay_transcript_path": str(
                _relative_to(active.repo_root, active.resolved_replay_transcript_path())
            ),
            "heldout_replay_path": str(
                _relative_to(active.repo_root, active.resolved_heldout_replay_path())
            ),
        },
        "field_principles": field_principles(),
        "tests_run": list(active.tests_run),
        "duration_s": _round(active.clock() - started),
    }
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load source artifacts and attach persisted final weights to the config."""

    exp3020_artifact = _read_json(config.resolved_exp3020_artifact_path())
    controller_config = _read_json(config.resolved_controller_config_path())
    if "final_weights" not in controller_config:
        summary = _mapping(exp3020_artifact.get("controller_summary"))
        controller_config["final_weights"] = dict(_mapping(summary.get("final_weights")))
    return SourceBundle(
        exp3032_artifact=_read_json(config.resolved_exp3032_artifact_path()),
        heldout_replay_rows=tuple(_read_jsonl(config.resolved_heldout_replay_path())),
        exp3020_artifact=exp3020_artifact,
        controller_config=controller_config,
        replay_transcript_rows=tuple(_read_jsonl(config.resolved_replay_transcript_path())),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first missing or unsafe source blocker, if any."""

    if not sources.exp3032_artifact:
        return "exp3032_artifact_missing_or_empty"
    if sources.exp3032_artifact.get("_malformed") is True:
        return "exp3032_artifact_malformed"
    if sources.exp3032_artifact.get("fr11_heldout_replay_ready") is not True:
        return "exp3032_heldout_replay_not_ready"
    if not sources.heldout_replay_rows:
        return "exp3032_heldout_replay_missing"
    if not sources.exp3020_artifact:
        return "exp3020_artifact_missing_or_empty"
    if sources.exp3020_artifact.get("_malformed") is True:
        return "exp3020_artifact_malformed"
    if sources.exp3020_artifact.get("verifier_feedback_controller_ready") is not True:
        return "exp3020_controller_not_ready"
    if sources.exp3020_artifact.get("native_llm_training_claim_made") is not False:
        return "exp3020_native_llm_training_claimed"
    if not _controller_weights(sources.controller_config):
        return "exp3020_controller_weights_missing"
    if not sources.replay_transcript_rows:
        return "exp3020_replay_transcript_missing"
    return None


def trace_precondition_blocker(
    prior_traces: Sequence[FeedbackTrace],
    heldout_traces: Sequence[FeedbackTrace],
) -> str | None:
    """Return a trace-level blocker after source files have loaded."""

    if not prior_traces:
        return "prior_exact_traces_missing"
    if not heldout_traces:
        return "heldout_feedback_traces_missing"
    if {trace.expected_feedback for trace in prior_traces} != {True, False}:
        return "prior_exact_trace_labels_unbalanced"
    if {trace.expected_feedback for trace in heldout_traces} != {True, False}:
        return "heldout_feedback_trace_labels_unbalanced"
    return None


def build_prior_exact_traces(sources: SourceBundle) -> list[FeedbackTrace]:
    """Extract prior machine-checked exact traces from the Exp 3020 transcript."""

    traces: list[FeedbackTrace] = []
    for row in sources.replay_transcript_rows:
        if row.get("partition") != "train":
            continue
        if row.get("exact_machine_checked") is not True:
            continue
        features = tuple(_string_list(row.get("features")))
        if not features:
            continue
        traces.append(
            FeedbackTrace(
                trace_id=f"prior::{row.get('row_id')}",
                source_experiment="exp3020",
                item_id=str(row.get("item_id") or ""),
                row_id=str(row.get("row_id") or ""),
                expected_feedback=row.get("exact_feedback") is True,
                exact_machine_checked=True,
                features=features,
            )
        )
    return traces


def build_heldout_feedback_traces(sources: SourceBundle) -> list[FeedbackTrace]:
    """Extract held-out exact feedback traces from the Exp 3032 replay table."""

    traces: list[FeedbackTrace] = []
    for row in sources.heldout_replay_rows:
        features = tuple(_string_list(row.get("checker_features")))
        exact_checked = bool(
            row.get("exact_claim_id") and _sequence(row.get("expected_authorities"))
        )
        if not features or not exact_checked:
            continue
        traces.append(
            FeedbackTrace(
                trace_id=str(row.get("trace_id") or f"heldout::{row.get('row_id')}"),
                source_experiment="exp3032",
                item_id=str(row.get("item_id") or ""),
                row_id=str(row.get("row_id") or ""),
                expected_feedback=row.get("expected_label") is True,
                exact_machine_checked=exact_checked,
                features=features,
            )
        )
    return traces


def stress_controller(
    prior_traces: Sequence[FeedbackTrace],
    heldout_traces: Sequence[FeedbackTrace],
    controller_config: Mapping[str, Any],
    *,
    retention_tolerance: float = RETENTION_TOLERANCE,
) -> JsonDict:
    """Apply held-out feedback and compute retention plus control deltas."""

    weights = _controller_weights(controller_config)
    updated = apply_feedback_updates(weights, heldout_traces, controller_config)
    shuffled_updated = apply_feedback_updates(
        weights,
        shuffled_feedback_traces(heldout_traces),
        controller_config,
    )
    irrelevant_updated = apply_feedback_updates(
        weights,
        adversarial_irrelevant_feedback_traces(heldout_traces),
        controller_config,
    )

    prior_before = retention_score(weights, prior_traces)
    prior_after = retention_score(updated, prior_traces)
    heldout_before = mean_signed_margin(weights, heldout_traces)
    heldout_after = mean_signed_margin(updated, heldout_traces)
    shuffled_after = mean_signed_margin(shuffled_updated, heldout_traces)
    irrelevant_after = mean_signed_margin(irrelevant_updated, heldout_traces)
    no_feedback_after = mean_signed_margin(weights, heldout_traces)

    prior_delta = _round(prior_after - prior_before)
    heldout_delta = _round(heldout_after - heldout_before)
    shuffled_delta = _round(shuffled_after - heldout_before)
    irrelevant_delta = _round(irrelevant_after - heldout_before)
    no_feedback_delta = _round(no_feedback_after - heldout_before)
    drift_failures = drift_failures_for(
        prior_traces,
        heldout_traces,
        controller_config,
        before_weights=weights,
        after_weights=updated,
        prior_retention_delta=prior_delta,
        heldout_delta_after_update=heldout_delta,
        shuffled_control_delta=shuffled_delta,
        adversarial_irrelevant_control_delta=irrelevant_delta,
        no_feedback_delta=no_feedback_delta,
        retention_tolerance=retention_tolerance,
    )
    return {
        "prior_retention_delta": prior_delta,
        "heldout_delta_after_update": heldout_delta,
        "shuffled_control_delta": shuffled_delta,
        "adversarial_irrelevant_control_delta": irrelevant_delta,
        "no_feedback_delta": no_feedback_delta,
        "drift_failures": drift_failures,
        "prior_retention_report": {
            "baseline_retention_score": prior_before,
            "after_update_retention_score": prior_after,
            "retention_tolerance": retention_tolerance,
            "prior_trace_count": len(prior_traces),
        },
        "heldout_update_report": {
            "baseline_signed_margin": heldout_before,
            "after_update_signed_margin": heldout_after,
            "heldout_trace_count": len(heldout_traces),
        },
        "control_report": {
            "shuffled_signed_margin": shuffled_after,
            "adversarial_irrelevant_signed_margin": irrelevant_after,
            "no_feedback_signed_margin": no_feedback_after,
            "negative_controls_rejected": bool(
                shuffled_delta <= 0.0 and irrelevant_delta <= 0.0 and no_feedback_delta == 0.0
            ),
        },
    }


def apply_feedback_updates(
    weights: Mapping[str, float],
    traces: Sequence[FeedbackTrace],
    controller_config: Mapping[str, Any],
) -> dict[str, float]:
    """Return bounded controller weights after replaying exact feedback traces."""

    proposed = dict(weights)
    learning_rate = float(controller_config.get("learning_rate", LEARNING_RATE))
    max_abs = float(controller_config.get("max_abs_weight", MAX_ABS_WEIGHT))
    allowed = tuple(_string_list(controller_config.get("allowed_feature_prefixes")))
    if not allowed:
        allowed = ALLOWED_FEATURE_PREFIXES
    for trace in traces:
        if not trace.exact_machine_checked:
            continue
        direction = 1.0 if trace.expected_feedback else -1.0
        for feature in trace.features:
            if not _feature_allowed(feature, allowed):
                continue
            value = proposed.get(feature, 0.0) + learning_rate * direction
            proposed[feature] = max(-max_abs, min(max_abs, value))
    return {key: _round(value) for key, value in sorted(proposed.items())}


def retention_score(weights: Mapping[str, float], traces: Sequence[FeedbackTrace]) -> float:
    """Return the fraction of prior exact traces still classified correctly."""

    if not traces:
        return 0.0
    retained = [signed_margin(weights, trace) >= 0.0 for trace in traces]
    return _round(sum(1.0 for item in retained if item) / len(retained))


def mean_signed_margin(
    weights: Mapping[str, float],
    traces: Sequence[FeedbackTrace],
) -> float:
    """Return average label-aligned margin so saturated classifiers can still move."""

    if not traces:
        return 0.0
    return _round(sum(signed_margin(weights, trace) for trace in traces) / len(traces))


def signed_margin(weights: Mapping[str, float], trace: FeedbackTrace) -> float:
    """Score a trace so positive values mean the controller agrees with feedback."""

    raw = sum(float(weights.get(feature, 0.0)) for feature in trace.features)
    return _round(raw if trace.expected_feedback else -raw)


def shuffled_feedback_traces(traces: Sequence[FeedbackTrace]) -> list[FeedbackTrace]:
    """Rotate labels across held-out rows for a deterministic leakage control."""

    if not traces:
        return []
    labels = [trace.expected_feedback for trace in traces]
    shifted = labels[1:] + labels[:1]
    return [
        FeedbackTrace(
            trace_id=f"shuffled::{trace.trace_id}",
            source_experiment="control",
            item_id=trace.item_id,
            row_id=trace.row_id,
            expected_feedback=label,
            exact_machine_checked=trace.exact_machine_checked,
            features=trace.features,
        )
        for trace, label in zip(traces, shifted, strict=True)
    ]


def adversarial_irrelevant_feedback_traces(
    traces: Sequence[FeedbackTrace],
) -> list[FeedbackTrace]:
    """Build off-task but syntactically allowed feedback that should not help."""

    return [
        FeedbackTrace(
            trace_id=f"irrelevant::{index}",
            source_experiment="control",
            item_id=f"irrelevant-{index}",
            row_id=f"irrelevant-{index}",
            expected_feedback=True,
            exact_machine_checked=True,
            features=(f"frontier::adversarial_irrelevant_feedback_{index}",),
        )
        for index, _trace in enumerate(traces)
    ]


def drift_failures_for(
    prior_traces: Sequence[FeedbackTrace],
    heldout_traces: Sequence[FeedbackTrace],
    controller_config: Mapping[str, Any],
    *,
    before_weights: Mapping[str, float],
    after_weights: Mapping[str, float],
    prior_retention_delta: float,
    heldout_delta_after_update: float,
    shuffled_control_delta: float,
    adversarial_irrelevant_control_delta: float,
    no_feedback_delta: float,
    retention_tolerance: float,
) -> list[str]:
    """Return visible failure strings for every gate that would block promotion."""

    failures: list[str] = []
    allowed = tuple(_string_list(controller_config.get("allowed_feature_prefixes")))
    if not allowed:
        allowed = ALLOWED_FEATURE_PREFIXES
    if prior_retention_delta < -retention_tolerance:
        failures.append(
            "prior_retention_below_tolerance:"
            f"{prior_retention_delta}<-{_round(retention_tolerance)}"
        )
    for trace in prior_traces:
        if (
            signed_margin(before_weights, trace) >= 0.0
            and signed_margin(after_weights, trace) < 0.0
        ):
            failures.append(f"prior_trace_forgotten:{trace.row_id}")
    if heldout_delta_after_update <= 0.0:
        failures.append(f"heldout_delta_not_positive:{heldout_delta_after_update}")
    if shuffled_control_delta > 0.0:
        failures.append(f"shuffled_control_improved:{shuffled_control_delta}")
    if adversarial_irrelevant_control_delta > 0.0:
        failures.append(
            f"adversarial_irrelevant_control_improved:{adversarial_irrelevant_control_delta}"
        )
    if no_feedback_delta != 0.0:
        failures.append(f"no_feedback_replay_moved:{no_feedback_delta}")
    for trace in (*prior_traces, *heldout_traces):
        for feature in trace.features:
            if feature in PROHIBITED_FEATURE_NAMES:
                failures.append(f"prohibited_feature:{trace.row_id}:{feature}")
            elif not _feature_allowed(feature, allowed):
                failures.append(f"feature_outside_controller_boundary:{trace.row_id}:{feature}")
    return sorted(dict.fromkeys(failures))


def _default_kan_helpers() -> Any:
    from carnot.eval import fr11_kan_cl_per_knot_self_learning_v1

    return fr11_kan_cl_per_knot_self_learning_v1


KAN_HELPERS_IMPORTER: Callable[[], Any] = _default_kan_helpers


def kan_locality_probe() -> JsonDict:
    """Run the existing KAN/RBF locality fixture if it is importable.

    This is intentionally separate from the FR-11 controller stress.  It only
    says whether a checked-in KAN-style local-update probe exists and whether
    that probe's own updates activate local centers; it does not turn the
    categorical FR-11 controller into a KAN.
    """

    try:
        helper = KAN_HELPERS_IMPORTER()
    except ImportError as exc:
        return {
            "available": False,
            "reason": f"missing_kan_helper:{exc}",
            "updates_concentrate_on_local_features": False,
        }
    try:
        stream = helper.build_constraint_stream(getattr(helper, "RANDOM_SEED", 2933))
        memory = helper.RBFImportanceMemory(centers=stream.centers)
        total_centers = int(stream.centers.shape[0])
        active_counts: list[int] = []
        off_rule_active = 0
        for rule in stream.rules:
            allowed = set(int(index) for index in rule.center_indices)
            for row in stream.train_by_constraint[rule.constraint_id]:
                activations = memory._activations(row.features)
                active_indices = {
                    int(index)
                    for index in (activations >= memory.active_threshold).nonzero()[0].tolist()
                }
                active_counts.append(len(active_indices))
                off_rule_active += len(active_indices - allowed)
        mean_active = sum(active_counts) / len(active_counts) if active_counts else 0.0
        mean_fraction = _round(mean_active / total_centers) if total_centers else 0.0
        concentrated = bool(active_counts and mean_fraction < 0.5 and off_rule_active == 0)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "available": False,
            "reason": f"kan_locality_probe_unusable:{exc}",
            "updates_concentrate_on_local_features": False,
        }
    return {
        "available": True,
        "source_module": "carnot.eval.fr11_kan_cl_per_knot_self_learning_v1",
        "locality_scope": "existing_seed_2933_rbf_per_center_fixture_not_fr11_controller",
        "total_centers": total_centers,
        "mean_active_centers_per_update": _round(mean_active),
        "mean_active_fraction": mean_fraction,
        "off_rule_active_center_count": off_rule_active,
        "updates_concentrate_on_local_features": concentrated,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the terminal artifact violates the Exp 3033 contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a dict")
    if substrate.get("live_llm_inference") is not False:
        raise ValueError("live_llm_inference must remain false")
    if substrate.get("model_weight_training") is not False:
        raise ValueError("model_weight_training must remain false")

    ready = artifact.get("fr11_nonforgetting_stress_ready") is True
    promotable = artifact.get("fr11_self_learning_promotable") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if not ready:
        if promotable:
            raise ValueError("blocked artifacts cannot be promotable")
        if not verdict.startswith(BLOCKED_PREFIXES):
            raise ValueError("honest_verdict must use blocked prefix")
        return
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal completion prefix")
    if promotable:
        if float(artifact.get("heldout_delta_after_update") or 0.0) <= 0.0:
            raise ValueError("heldout_delta_after_update must be positive")
        if float(artifact.get("shuffled_control_delta") or 0.0) > 0.0:
            raise ValueError("shuffled_control_delta must not improve")
        if float(artifact.get("adversarial_irrelevant_control_delta") or 0.0) > 0.0:
            raise ValueError("adversarial_irrelevant_control_delta must not improve")
        if float(artifact.get("no_feedback_delta") or 0.0) != 0.0:
            raise ValueError("no_feedback_delta must be exactly zero")
        tolerance = float(artifact.get("retention_tolerance", RETENTION_TOLERANCE))
        if float(artifact.get("prior_retention_delta") or 0.0) < -tolerance:
            raise ValueError("prior_retention_delta is below tolerance")
        if artifact.get("drift_failures") != []:
            raise ValueError("drift_failures must be empty for promotion")


def inference_substrate(*, controller_weight_update: bool) -> JsonDict:
    """Describe the execution substrate without overstating learning scope."""

    return {
        "mode": "cached_feedback_controller_stress",
        "live_llm_inference": False,
        "model_weight_training": False,
        "controller_weight_update": controller_weight_update,
        "cached_artifacts_only": True,
        "training_scope": "bounded_controller_weights_only",
    }


def field_principles() -> JsonDict:
    """Return compact reasons for the required terminal fields."""

    return {
        "fr11_nonforgetting_stress_ready": "Stress result must be complete before capstone.",
        "fr11_self_learning_promotable": "FR-11 promotion must be explicit and machine-readable.",
        "prior_retention_delta": "Continuous learning must not erase prior skills.",
        "heldout_delta_after_update": "New feedback must improve held-out behavior.",
        "shuffled_control_delta": "Gains must not come from label leakage.",
        "no_feedback_delta": "Update effect must be separated from replay noise.",
        "drift_failures": "Regressions must remain visible.",
        "kan_locality_probe_available": "KAN claims require actual local-probe evidence.",
        "inference_substrate": "Cached-feedback experiments must not claim live LLM inference.",
        "honest_verdict": "Terminal verdict must be machine-readable.",
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    """CLI entrypoint for focused Exp 3033 runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--stress-report", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(output_path=args.output, stress_report_path=args.stress_report)
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["fr11_nonforgetting_stress_ready"] else 1


def _stress_report_rows(stress: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {"section": "prior_retention", **_mapping(stress.get("prior_retention_report"))},
        {"section": "heldout_update", **_mapping(stress.get("heldout_update_report"))},
        {"section": "negative_controls", **_mapping(stress.get("control_report"))},
    ]


def _blocked_artifact(config: ExperimentConfig, duration_s: float, reason: str) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_nonforgetting_stress_ready": False,
        "fr11_self_learning_promotable": False,
        "prior_retention_delta": 0.0,
        "heldout_delta_after_update": 0.0,
        "shuffled_control_delta": 0.0,
        "adversarial_irrelevant_control_delta": 0.0,
        "no_feedback_delta": 0.0,
        "drift_failures": [reason],
        "kan_locality_probe_available": False,
        "kan_locality_report": {
            "available": False,
            "reason": "blocked_before_locality_probe",
            "updates_concentrate_on_local_features": False,
        },
        "inference_substrate": inference_substrate(controller_weight_update=False),
        "honest_verdict": f"blocked_{reason}",
        "blocked_reason": reason,
        "promotion_decision": "blocked",
        "retention_tolerance": config.retention_tolerance,
        "stress_report_path": str(
            _relative_to(config.repo_root, config.resolved_stress_report_path())
        ),
        "duration_s": duration_s,
        "tests_run": list(config.tests_run),
        "field_principles": field_principles(),
    }


def _controller_weights(controller_config: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(feature): float(weight)
        for feature, weight in _mapping(controller_config.get("final_weights")).items()
    }


def _feature_allowed(feature: str, allowed_prefixes: Sequence[str]) -> bool:
    return bool(
        feature not in PROHIBITED_FEATURE_NAMES and feature.startswith(tuple(allowed_prefixes))
    )


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {"_malformed": True}
    return dict(payload) if isinstance(payload, Mapping) else {"_malformed": True}


def _read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError:  # pragma: no cover - filesystem race/permission defense.
        return [{"_malformed": True}]
    rows: list[JsonDict] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return [{"_malformed": True}]
        if not isinstance(payload, Mapping):
            return [{"_malformed": True}]
        rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value)]


def _round(value: float) -> float:
    return round(float(value), 6)


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


if __name__ == "__main__":  # pragma: no cover - direct module execution.
    raise SystemExit(main())
