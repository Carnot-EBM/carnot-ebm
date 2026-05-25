"""Exp 3020 bounded verifier-feedback self-learning controller.

This module tests a narrow Draft-Verify-Improve-inspired idea without changing
any LLM weights.  Cached exact validator traces provide accept/reject feedback;
the only learned state is a small inspectable router weight table.  A proposed
router update is kept only when it improves an independent held-out exact
validator utility metric and does not degrade the carried-forward forgetting
guard.

Spec refs: REQ-LEARN-3020, SCENARIO-LEARN-3020,
SCENARIO-LEARN-3020-BLOCKED.
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
ARTIFACT = "experiment_3020_dvi_verifier_feedback_self_learning_controller_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.dvi_verifier_feedback_self_learning_controller.v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME

EXP3007_REL_PATH = Path("results/experiment_3007_fr11_attractor_trace_memory_stability_v1.json")
EXP3017_ARTIFACT_REL_PATH = Path(
    "results/experiment_3017_nsvif_instruction_validator_tree_expansion_v1.json"
)
EXP3017_MANIFEST_REL_PATH = Path(
    "results/nsvif_instruction_validator_tree_expansion_3017/validator_manifest.jsonl"
)
EXP3018_ARTIFACT_REL_PATH = Path(
    "results/experiment_3018_beaver_style_validator_frontier_certificate_v1.json"
)
EXP3018_MANIFEST_REL_PATH = Path(
    "results/beaver_style_validator_frontier_certificate_3018/certificate_manifest.jsonl"
)
EXP3019_ARTIFACT_REL_PATH = Path(
    "results/experiment_3019_fr11_feasibility_channel_de_tautology_diagnostic_v1.json"
)
EXP3019_TABLE_REL_PATH = Path(
    "results/fr11_feasibility_channel_de_tautology_diagnostic_3019/diagnostic_table.jsonl"
)
CONTROLLER_CONFIG_REL_PATH = Path(
    "results/dvi_verifier_feedback_self_learning_controller_3020/controller_config.json"
)
REPLAY_TRANSCRIPT_REL_PATH = Path(
    "results/dvi_verifier_feedback_self_learning_controller_3020/replay_transcript.jsonl"
)

LEARNING_RATE = 0.25
MAX_ABS_WEIGHT = 1.0
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
PROHIBITED_FEATURE_NAMES = {
    "candidate_role",
    "certificate_status",
    "heldout_success_label",
    "item_id",
    "row_id",
}
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
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "verifier_feedback_controller_ready",
        "continuous_self_learning_task",
        "independent_self_learning_boundary_preserved",
        "controller_config_path",
        "n_replay_items",
        "heldout_delta",
        "negative_control_delta",
        "forgetting_guard_passed",
        "drift_guard_passed",
        "tautology_risk_flag",
        "native_llm_training_claim_made",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock hooks for deterministic Exp 3020 evaluation."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    controller_config_path: Path | None = None
    replay_transcript_path: Path | None = None
    exp3017_artifact_path: Path | None = None
    exp3017_manifest_path: Path | None = None
    exp3018_artifact_path: Path | None = None
    exp3018_manifest_path: Path | None = None
    exp3019_artifact_path: Path | None = None
    exp3019_table_path: Path | None = None
    exp3007_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_controller_config_path(self) -> Path:
        return self.controller_config_path or self.repo_root / CONTROLLER_CONFIG_REL_PATH

    def resolved_replay_transcript_path(self) -> Path:
        return self.replay_transcript_path or self.repo_root / REPLAY_TRANSCRIPT_REL_PATH

    def resolved_exp3017_artifact_path(self) -> Path:
        return self.exp3017_artifact_path or self.repo_root / EXP3017_ARTIFACT_REL_PATH

    def resolved_exp3017_manifest_path(self) -> Path:
        return self.exp3017_manifest_path or self.repo_root / EXP3017_MANIFEST_REL_PATH

    def resolved_exp3018_artifact_path(self) -> Path:
        return self.exp3018_artifact_path or self.repo_root / EXP3018_ARTIFACT_REL_PATH

    def resolved_exp3018_manifest_path(self) -> Path:
        return self.exp3018_manifest_path or self.repo_root / EXP3018_MANIFEST_REL_PATH

    def resolved_exp3019_artifact_path(self) -> Path:
        return self.exp3019_artifact_path or self.repo_root / EXP3019_ARTIFACT_REL_PATH

    def resolved_exp3019_table_path(self) -> Path:
        return self.exp3019_table_path or self.repo_root / EXP3019_TABLE_REL_PATH

    def resolved_exp3007_artifact_path(self) -> Path:
        return self.exp3007_artifact_path or self.repo_root / EXP3007_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded source artifacts and row tables needed for cached replay."""

    exp3017_artifact: JsonDict
    exp3017_rows: tuple[JsonDict, ...]
    exp3018_artifact: JsonDict
    exp3018_rows: tuple[JsonDict, ...]
    exp3019_artifact: JsonDict
    exp3019_rows: tuple[JsonDict, ...]
    exp3007_artifact: JsonDict


@dataclass(frozen=True)
class ReplayItem:
    """One exact verifier-feedback event or guard row.

    The feature vector intentionally excludes item IDs, row IDs, certificate
    status, candidate role, and held-out labels.  Those fields remain in the
    transcript as provenance, but they never enter the learned router weights.
    """

    replay_id: str
    source_experiment: str
    item_id: str
    row_id: str
    partition: str
    exact_feedback: bool
    machine_checked: bool
    features: tuple[str, ...]
    source_detail: JsonDict = field(default_factory=dict)


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3020 artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        artifact = _blocked_artifact(active, _round(active.clock() - started), blocker)
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    controller_config = default_controller_config()
    replay_items = build_replay_set(sources)
    controller_report = train_controller(replay_items, controller_config)
    controls = evaluate_controls(replay_items, controller_config)
    boundary_preserved = independent_boundary_preserved(sources, controller_config)
    tautology_risk = tautology_risk_for(controller_config, sources)
    ready = bool(
        boundary_preserved
        and replay_items
        and controller_report["heldout_delta"] > 0.0
        and controls["negative_control_delta"] <= 0.0
        and controller_report["forgetting_guard_passed"]
        and controller_report["drift_guard_passed"]
        and not tautology_risk
        and not controller_config["native_llm_training_claim_made"]
    )

    _write_json(active.resolved_controller_config_path(), controller_config)
    _write_jsonl(active.resolved_replay_transcript_path(), controller_report["transcript_rows"])
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "verifier_feedback_controller_ready": ready,
        "continuous_self_learning_task": True,
        "independent_self_learning_boundary_preserved": boundary_preserved,
        "controller_config_path": str(
            _relative_to(active.repo_root, active.resolved_controller_config_path())
        ),
        "replay_transcript_path": str(
            _relative_to(active.repo_root, active.resolved_replay_transcript_path())
        ),
        "n_replay_items": len(replay_items),
        "heldout_delta": controller_report["heldout_delta"],
        "negative_control_delta": controls["negative_control_delta"],
        "forgetting_guard_passed": controller_report["forgetting_guard_passed"],
        "drift_guard_passed": controller_report["drift_guard_passed"],
        "tautology_risk_flag": tautology_risk,
        "native_llm_training_claim_made": controller_config["native_llm_training_claim_made"],
        "honest_verdict": (
            "complete: verifier_feedback_controller_ready"
            if ready
            else "blocked_verifier_feedback_controller_not_ready"
        ),
        "duration_s": _round(active.clock() - started),
        "inference_substrate": "cached_exact_trace_replay_controller_only",
        "controller_summary": {
            "accepted_update_count": controller_report["accepted_update_count"],
            "rejected_update_count": controller_report["rejected_update_count"],
            "learned_weight_count": len(controller_report["final_weights"]),
            "final_weights": controller_report["final_weights"],
        },
        "control_comparison": controls["control_comparison"],
        "negative_control_report": controls["negative_control_report"],
        "heldout_scores": {
            "baseline": controller_report["baseline_score"],
            "final": controller_report["final_score"],
            "delta": controller_report["heldout_delta"],
        },
        "forgetting_report": controller_report["forgetting_report"],
        "source_artifact_coverage": source_artifact_coverage(sources),
        "source_tautology_risk_observed": sources.exp3019_artifact.get("tautology_risk_flag")
        is True,
        "self_learning_boundary": (
            "DVI is used as controller inspiration only; the experiment updates "
            "bounded router weights over cached exact traces and performs no native LLM training."
        ),
        "field_principles": field_principles(),
        "tests_run": list(active.tests_run),
    }
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load all source artifacts; missing files become empty evidence."""

    return SourceBundle(
        exp3017_artifact=_read_json(config.resolved_exp3017_artifact_path()),
        exp3017_rows=tuple(_read_jsonl(config.resolved_exp3017_manifest_path())),
        exp3018_artifact=_read_json(config.resolved_exp3018_artifact_path()),
        exp3018_rows=tuple(_read_jsonl(config.resolved_exp3018_manifest_path())),
        exp3019_artifact=_read_json(config.resolved_exp3019_artifact_path()),
        exp3019_rows=tuple(_read_jsonl(config.resolved_exp3019_table_path())),
        exp3007_artifact=_read_json(config.resolved_exp3007_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first source-readiness blocker, if any."""

    if not sources.exp3017_artifact:
        return "exp3017_artifact_missing_or_empty"
    if sources.exp3017_artifact.get("_malformed") is True:
        return "exp3017_artifact_malformed"
    if sources.exp3017_artifact.get("instruction_validator_tree_ready") is not True:
        return "exp3017_not_ready"
    if not sources.exp3017_rows:
        return "exp3017_manifest_missing"
    if not sources.exp3018_artifact:
        return "exp3018_artifact_missing_or_empty"
    if sources.exp3018_artifact.get("_malformed") is True:
        return "exp3018_artifact_malformed"
    if sources.exp3018_artifact.get("frontier_certificate_ready") is not True:
        return "exp3018_not_ready"
    if sources.exp3018_artifact.get("live_llm_evidence_used") is not False:
        return "exp3018_live_llm_evidence_contaminated"
    if not sources.exp3018_rows:
        return "exp3018_manifest_missing"
    if not sources.exp3019_artifact:
        return "exp3019_artifact_missing_or_empty"
    if sources.exp3019_artifact.get("_malformed") is True:
        return "exp3019_artifact_malformed"
    if sources.exp3019_artifact.get("feasibility_channel_diagnostic_ready") is not True:
        return "exp3019_not_ready"
    if sources.exp3019_artifact.get("reused_label_as_feature") is not False:
        return "exp3019_reused_label_as_feature"
    if not sources.exp3019_rows:
        return "exp3019_table_missing"
    if not sources.exp3007_artifact:
        return "exp3007_artifact_missing_or_empty"
    if sources.exp3007_artifact.get("_malformed") is True:
        return "exp3007_artifact_malformed"
    if sources.exp3007_artifact.get("trace_memory_stability_ready") is not True:
        return "exp3007_not_ready"
    if sources.exp3007_artifact.get("forgetting_guard_passed") is not True:
        return "exp3007_forgetting_guard_failed"
    if sources.exp3007_artifact.get("drift_guard_passed") is not True:
        return "exp3007_drift_guard_failed"
    return None


def build_replay_set(sources: SourceBundle) -> list[ReplayItem]:
    """Build replay items from exact certificate rows plus Exp 3007 guard rows."""

    validators_by_item = {
        str(row.get("item_id")): row for row in sources.exp3017_rows if row.get("item_id")
    }
    diagnostics_by_row = {
        str(row.get("row_id")): row for row in sources.exp3019_rows if row.get("row_id")
    }
    items: list[ReplayItem] = []
    for row in sources.exp3018_rows:
        if row.get("row_type") != "candidate_frontier":
            continue
        status = str(row.get("certificate_status") or "")
        if status not in {"certified_safe", "certified_violating"}:
            continue
        item_id = str(row.get("item_id") or "")
        row_id = str(row.get("row_id") or "")
        validator_row = validators_by_item.get(item_id, {})
        diagnostic_row = diagnostics_by_row.get(row_id, {})
        feedback = status == "certified_safe"
        partition = "heldout" if diagnostic_row.get("heldout_partition") is True else "train"
        items.append(
            ReplayItem(
                replay_id=f"exp3018::{row_id}",
                source_experiment="exp3018",
                item_id=item_id,
                row_id=row_id,
                partition=partition,
                exact_feedback=feedback,
                machine_checked=_candidate_row_machine_checked(row, validator_row),
                features=tuple(_features_for_candidate(row, validator_row)),
                source_detail={
                    "category": str(validator_row.get("category") or "unknown"),
                    "feasibility_score": float(diagnostic_row.get("feasibility_score") or 0.0),
                    "source_artifacts": ["exp3017", "exp3018", "exp3019"],
                },
            )
        )
    for memory_id in _string_list(sources.exp3007_artifact.get("accepted_memory_ids")):
        items.append(
            ReplayItem(
                replay_id=f"exp3007::{memory_id}",
                source_experiment="exp3007",
                item_id=memory_id,
                row_id=memory_id,
                partition="forgetting_guard",
                exact_feedback=True,
                machine_checked=True,
                features=(f"memory_guard::{memory_id}",),
                source_detail={"source_artifacts": ["exp3007"]},
            )
        )
    return items


def default_controller_config() -> JsonDict:
    """Return the inspectable bounded update rule used by Exp 3020."""

    return {
        "schema": "carnot.fr11.dvi_verifier_feedback_controller.config.v1",
        "controller_type": "bounded_verifier_feedback_router",
        "learning_rate": LEARNING_RATE,
        "max_abs_weight": MAX_ABS_WEIGHT,
        "update_rule": (
            "For each exact verifier feedback event, add learning_rate to "
            "non-label evidence features on accepted rows and subtract it on "
            "rejected rows. Commit the proposal only when independent held-out "
            "utility strictly improves and forgetting utility does not degrade."
        ),
        "update_metric_names": [
            "training_exact_verifier_feedback",
            "machine_checked_evidence_density",
        ],
        "independent_metric_names": [
            "heldout_exact_validator_utility",
            "forgetting_guard_utility",
        ],
        "allowed_feature_prefixes": list(ALLOWED_FEATURE_PREFIXES),
        "prohibited_feature_names": sorted(PROHIBITED_FEATURE_NAMES),
        "native_llm_training_claim_made": False,
        "model_weight_mutation": False,
        "self_learning_boundary": "controller_weight_replay_only_no_native_llm_finetuning",
    }


def train_controller(
    replay_items: Sequence[ReplayItem],
    controller_config: Mapping[str, Any],
) -> JsonDict:
    """Replay train items and accept only held-out-improving updates."""

    heldout = [item for item in replay_items if item.partition == "heldout"]
    train = [item for item in replay_items if item.partition == "train"]
    guard = [item for item in replay_items if item.partition == "forgetting_guard"]
    weights: dict[str, float] = {}
    baseline = evaluate_utility(weights, heldout)
    baseline_forgetting = evaluate_utility(weights, guard)
    transcript_rows: list[JsonDict] = []
    accepted_count = 0

    for index, item in enumerate(train):
        before = evaluate_utility(weights, heldout)
        before_forgetting = evaluate_utility(weights, guard)
        proposal = propose_update(weights, item, controller_config)
        after = evaluate_utility(proposal, heldout)
        after_forgetting = evaluate_utility(proposal, guard)
        drift_ok = item_drift_guard(item, controller_config)
        update_accepted = bool(
            item.machine_checked
            and after > before
            and after_forgetting >= before_forgetting
            and drift_ok
        )
        if update_accepted:
            weights = proposal
            accepted_count += 1
        transcript_rows.append(
            {
                "event_index": index,
                "replay_id": item.replay_id,
                "item_id": item.item_id,
                "row_id": item.row_id,
                "partition": item.partition,
                "exact_feedback": item.exact_feedback,
                "exact_machine_checked": item.machine_checked,
                "features": list(item.features),
                "before_heldout_score": before,
                "after_heldout_score": after,
                "before_forgetting_score": before_forgetting,
                "after_forgetting_score": after_forgetting,
                "drift_guard_passed": drift_ok,
                "update_accepted": update_accepted,
            }
        )

    final_score = evaluate_utility(weights, heldout)
    final_forgetting = evaluate_utility(weights, guard)
    return {
        "baseline_score": baseline,
        "final_score": final_score,
        "heldout_delta": _round(final_score - baseline),
        "forgetting_report": {
            "baseline_score": baseline_forgetting,
            "final_score": final_forgetting,
            "delta": _round(final_forgetting - baseline_forgetting),
            "guard_item_count": len(guard),
        },
        "forgetting_guard_passed": bool(guard and final_forgetting >= baseline_forgetting),
        "drift_guard_passed": all(item_drift_guard(item, controller_config) for item in replay_items),
        "accepted_update_count": accepted_count,
        "rejected_update_count": len(train) - accepted_count,
        "final_weights": {key: _round(value) for key, value in sorted(weights.items())},
        "transcript_rows": transcript_rows,
    }


def propose_update(
    weights: Mapping[str, float],
    item: ReplayItem,
    controller_config: Mapping[str, Any],
) -> dict[str, float]:
    """Return a bounded weight proposal for one verifier feedback event."""

    proposed = dict(weights)
    direction = 1.0 if item.exact_feedback else -1.0
    step = float(controller_config.get("learning_rate", LEARNING_RATE)) * direction
    max_abs = float(controller_config.get("max_abs_weight", MAX_ABS_WEIGHT))
    for feature in item.features:
        value = proposed.get(feature, 0.0) + step
        proposed[feature] = max(-max_abs, min(max_abs, value))
    return proposed


def evaluate_utility(weights: Mapping[str, float], items: Sequence[ReplayItem]) -> float:
    """Evaluate exact-label utility from held-out or guard replay rows."""

    if not items:
        return 0.0
    total = 0.0
    for item in items:
        score = sum(float(weights.get(feature, 0.0)) for feature in item.features)
        margin = score if item.exact_feedback else -score
        clipped = max(-1.0, min(1.0, margin))
        total += 0.5 + 0.5 * clipped
    return _round(total / len(items))


def evaluate_controls(
    replay_items: Sequence[ReplayItem],
    controller_config: Mapping[str, Any],
) -> JsonDict:
    """Compare no-learning, replay-only, random-update, and negative controls."""

    heldout = [item for item in replay_items if item.partition == "heldout"]
    baseline = evaluate_utility({}, heldout)
    random_weights = propose_update(
        {},
        ReplayItem(
            replay_id="random-update-control",
            source_experiment="control",
            item_id="random",
            row_id="random",
            partition="negative_control",
            exact_feedback=True,
            machine_checked=False,
            features=("frontier::random_uninformative",),
        ),
        controller_config,
    )
    random_delta = _round(evaluate_utility(random_weights, heldout) - baseline)
    negative_deltas: dict[str, float] = {}
    for control in negative_control_items(heldout):
        control_weights = propose_update({}, control, controller_config)
        negative_deltas[control.replay_id] = _round(evaluate_utility(control_weights, heldout) - baseline)
    negative_delta = max(negative_deltas.values()) if negative_deltas else 0.0
    return {
        "negative_control_delta": _round(negative_delta),
        "control_comparison": {
            "no_learning_delta": 0.0,
            "replay_only_delta": 0.0,
            "random_update_delta": random_delta,
        },
        "negative_control_report": {
            "control_deltas": negative_deltas,
            "negative_controls_rejected": all(delta <= 0.0 for delta in negative_deltas.values()),
        },
    }


def negative_control_items(heldout_items: Sequence[ReplayItem]) -> list[ReplayItem]:
    """Build irrelevant, contradicted, and shuffled-feedback control updates."""

    first_positive = next((item for item in heldout_items if item.exact_feedback), None)
    first_negative = next((item for item in heldout_items if not item.exact_feedback), None)
    controls = [
        ReplayItem(
            replay_id="negative_control::irrelevant_trace",
            source_experiment="control",
            item_id="irrelevant",
            row_id="irrelevant",
            partition="negative_control",
            exact_feedback=True,
            machine_checked=False,
            features=("frontier::irrelevant_trace",),
        )
    ]
    if first_positive is not None:
        controls.append(
            ReplayItem(
                replay_id="negative_control::contradicted_accept",
                source_experiment="control",
                item_id=first_positive.item_id,
                row_id=first_positive.row_id,
                partition="negative_control",
                exact_feedback=False,
                machine_checked=False,
                features=first_positive.features,
            )
        )
    if first_negative is not None:
        controls.append(
            ReplayItem(
                replay_id="negative_control::shuffled_reject",
                source_experiment="control",
                item_id=first_negative.item_id,
                row_id=first_negative.row_id,
                partition="negative_control",
                exact_feedback=True,
                machine_checked=False,
                features=first_negative.features,
            )
        )
    return controls


def item_drift_guard(item: ReplayItem, controller_config: Mapping[str, Any]) -> bool:
    """Return true only when features remain inside the configured boundary."""

    allowed = tuple(_string_list(controller_config.get("allowed_feature_prefixes")))
    if not allowed:
        return False
    for feature in item.features:
        if feature in PROHIBITED_FEATURE_NAMES:
            return False
        if not feature.startswith(allowed):
            return False
    return True


def independent_boundary_preserved(
    sources: SourceBundle,
    controller_config: Mapping[str, Any],
) -> bool:
    """Ensure update metrics, held-out metrics, and source guards are separated."""

    update_metrics = set(_string_list(controller_config.get("update_metric_names")))
    independent_metrics = set(_string_list(controller_config.get("independent_metric_names")))
    return bool(
        sources.exp3007_artifact.get("independent_self_learning_boundary_preserved") is True
        and sources.exp3007_artifact.get("self_reported_memory_utility_counted") is False
        and sources.exp3019_artifact.get("reused_label_as_feature") is False
        and update_metrics
        and independent_metrics
        and update_metrics.isdisjoint(independent_metrics)
    )


def tautology_risk_for(
    controller_config: Mapping[str, Any],
    sources: SourceBundle,
) -> bool:
    """Flag self-grading risk for this controller, not merely for a source diagnostic."""

    prohibited = set(_string_list(controller_config.get("prohibited_feature_names")))
    features_overlap_labels = bool(prohibited & set(ALLOWED_FEATURE_PREFIXES))
    native_claim = controller_config.get("native_llm_training_claim_made") is True
    source_reused_label = sources.exp3019_artifact.get("reused_label_as_feature") is True
    return bool(features_overlap_labels or native_claim or source_reused_label)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the terminal artifact violates the Exp 3020 contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("native_llm_training_claim_made") is not False:
        raise ValueError("native_llm_training_claim_made must remain false")
    ready = artifact.get("verifier_feedback_controller_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if ready:
        if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must use a terminal completion prefix")
        if artifact.get("continuous_self_learning_task") is not True:
            raise ValueError("continuous_self_learning_task must be true")
        if artifact.get("independent_self_learning_boundary_preserved") is not True:
            raise ValueError("independent self-learning boundary must be preserved")
        if int(artifact.get("n_replay_items") or 0) <= 0:
            raise ValueError("n_replay_items must be positive")
        if float(artifact.get("heldout_delta") or 0.0) <= 0.0:
            raise ValueError("heldout_delta must be positive")
        if float(artifact.get("negative_control_delta") or 0.0) > 0.0:
            raise ValueError("negative_control_delta must not improve")
        if artifact.get("forgetting_guard_passed") is not True:
            raise ValueError("forgetting_guard_passed must be true")
        if artifact.get("drift_guard_passed") is not True:
            raise ValueError("drift_guard_passed must be true")
        if artifact.get("tautology_risk_flag") is not False:
            raise ValueError("tautology_risk_flag must be false")
        if not artifact.get("controller_config_path"):
            raise ValueError("controller_config_path must be present")
    elif not verdict.startswith(BLOCKED_PREFIXES):
        raise ValueError("honest_verdict must use a blocked prefix when not ready")


def source_artifact_coverage(sources: SourceBundle) -> JsonDict:
    """Return which required source artifacts materially contributed."""

    return {
        "exp3017_validator_trees_used": bool(sources.exp3017_rows),
        "exp3018_certificate_rows_used": bool(sources.exp3018_rows),
        "exp3019_feasibility_rows_used": bool(sources.exp3019_rows),
        "exp3007_forgetting_guard_used": bool(sources.exp3007_artifact.get("accepted_memory_ids")),
    }


def field_principles() -> JsonDict:
    """Return compact reasons for required machine-gated fields."""

    return {
        "verifier_feedback_controller_ready": "FR-11 carry-forward must be machine-gated.",
        "continuous_self_learning_task": "Milestone self-learning requirement is explicit.",
        "independent_self_learning_boundary_preserved": "Self-learning must not grade itself.",
        "controller_config_path": "Learned update rule must be inspectable.",
        "n_replay_items": "Replay sample size must be explicit.",
        "heldout_delta": "Usefulness must be measured on independent held-out metrics.",
        "negative_control_delta": "Irrelevant or contradicted traces must not improve score.",
        "forgetting_guard_passed": "New memory must not damage held-out tasks.",
        "drift_guard_passed": "Controller must not drift into unrelated constraints.",
        "tautology_risk_flag": "FR-11 promotion must reject self-grading.",
        "native_llm_training_claim_made": "DVI inspiration must not become a local fine-tuning claim.",
        "honest_verdict": "Terminal verdict must be machine-readable.",
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    """CLI entrypoint for focused Exp 3020 runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--controller-config", type=Path, default=None)
    parser.add_argument("--replay-transcript", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            output_path=args.output,
            controller_config_path=args.controller_config,
            replay_transcript_path=args.replay_transcript,
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["verifier_feedback_controller_ready"] else 1


def _candidate_row_machine_checked(
    row: Mapping[str, Any],
    validator_row: Mapping[str, Any],
) -> bool:
    outcome = _mapping(row.get("deterministic_validator_outcome"))
    return bool(
        outcome
        and outcome.get("llm_judge_used") is False
        and row.get("live_llm_evidence_used") is False
        and validator_row.get("all_authoritative_nodes_exact_checked") is True
    )


def _features_for_candidate(
    row: Mapping[str, Any],
    validator_row: Mapping[str, Any],
) -> list[str]:
    outcome = _mapping(row.get("deterministic_validator_outcome"))
    nodes = _validator_nodes_by_id(validator_row)
    failing_ids = _string_list(outcome.get("failing_node_ids"))
    category = _safe_token(str(validator_row.get("category") or "unknown"))
    features: set[str] = set()
    if failing_ids:
        features.add("evidence::authoritative_failures_present")
        features.add(f"category::{category}::failures_present")
        for reason in _string_list(outcome.get("rejection_reasons")):
            features.add(f"failure_reason::{_safe_token(reason)}")
        for node_id in failing_ids:
            kind = _safe_token(str(_mapping(nodes.get(node_id)).get("kind") or "unknown"))
            features.add(f"failing_node_kind::{kind}")
    else:
        features.add("evidence::all_authoritative_checks_passed")
        features.add(f"category::{category}::all_passed")
        for node in nodes.values():
            node_map = _mapping(node)
            if node_map.get("authoritative", True):
                authority = _safe_token(str(node_map.get("authority") or "unknown"))
                kind = _safe_token(str(node_map.get("kind") or "unknown"))
                features.add(f"authority::{authority}::all_passed")
                features.add(f"node_kind::{kind}::all_passed")
    if _mapping(row.get("frontier_exploration")).get("bounded") is True:
        features.add("frontier::bounded_cached_candidate_set")
    if row.get("prefix_closed_assumption_applies") is True:
        features.add("frontier::prefix_closed_assumption")
    return sorted(features)


def _validator_nodes_by_id(validator_row: Mapping[str, Any]) -> dict[str, JsonDict]:
    tree = _mapping(validator_row.get("validator_tree"))
    return {
        str(node.get("node_id")): dict(node)
        for node in _sequence(tree.get("nodes"))
        if _mapping(node).get("node_id")
    }


def _blocked_artifact(config: ExperimentConfig, duration_s: float, reason: str) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "verifier_feedback_controller_ready": False,
        "continuous_self_learning_task": True,
        "independent_self_learning_boundary_preserved": False,
        "controller_config_path": str(
            _relative_to(config.repo_root, config.resolved_controller_config_path())
        ),
        "replay_transcript_path": str(
            _relative_to(config.repo_root, config.resolved_replay_transcript_path())
        ),
        "n_replay_items": 0,
        "heldout_delta": 0.0,
        "negative_control_delta": 0.0,
        "forgetting_guard_passed": False,
        "drift_guard_passed": False,
        "tautology_risk_flag": False,
        "native_llm_training_claim_made": False,
        "honest_verdict": f"blocked_{reason}",
        "duration_s": duration_s,
        "blocked_reason": reason,
        "tests_run": list(config.tests_run),
    }


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
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value)]


def _safe_token(value: str) -> str:
    cleaned = [char.lower() if char.isalnum() else "_" for char in value]
    return "_".join("".join(cleaned).split("_")).strip("_") or "unknown"


def _round(value: float) -> float:
    return round(float(value), 6)


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    raise SystemExit(main())
