"""Exp 3032 held-out DVI verifier-feedback replay.

This module replays the bounded Exp 3020 verifier-feedback controller on exact
held-out verifier traces.  It deliberately stays in cached-data territory: no
live LLM call is made, no model weights are trained, and the only learned state
being inspected is the small controller weight table produced by Exp 3020.

Spec refs: REQ-LEARN-3032, SCENARIO-LEARN-3032,
SCENARIO-LEARN-3032-BOUNDED, SCENARIO-LEARN-3032-BLOCKED.
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
ARTIFACT = "experiment_3032_fr11_heldout_dvi_replay_v2"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.heldout_dvi_replay.v2"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME

EXP3019_ARTIFACT_REL_PATH = Path(
    "results/experiment_3019_fr11_feasibility_channel_de_tautology_diagnostic_v1.json"
)
EXP3019_TABLE_REL_PATH = Path(
    "results/fr11_feasibility_channel_de_tautology_diagnostic_3019/diagnostic_table.jsonl"
)
EXP3020_ARTIFACT_REL_PATH = Path(
    "results/experiment_3020_dvi_verifier_feedback_self_learning_controller_v1.json"
)
CONTROLLER_CONFIG_REL_PATH = Path(
    "results/dvi_verifier_feedback_self_learning_controller_3020/controller_config.json"
)
REPLAY_TRANSCRIPT_REL_PATH = Path(
    "results/dvi_verifier_feedback_self_learning_controller_3020/replay_transcript.jsonl"
)
EXP3017_MANIFEST_REL_PATH = Path(
    "results/nsvif_instruction_validator_tree_expansion_3017/validator_manifest.jsonl"
)
EXP3018_MANIFEST_REL_PATH = Path(
    "results/beaver_style_validator_frontier_certificate_3018/certificate_manifest.jsonl"
)
HELDOUT_REPLAY_REL_PATH = Path("results/fr11_heldout_dvi_replay_3032/heldout_replay.jsonl")

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
PROHIBITED_CHECKER_FIELDS = frozenset(
    {
        "candidate_role",
        "certificate_status",
        "heldout_success_label",
        "original_controller_rationale",
        "update_accepted",
    }
)
AUTHORITY_WITHHELD_PREFIXES = (
    "authority::",
    "category::",
    "evidence::",
    "failing_node_kind::",
    "failure_reason::",
    "node_kind::",
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "fr11_heldout_replay_ready",
        "continuous_self_learning_tested",
        "heldout_trace_count",
        "feasible_infeasible_auc_delta",
        "shuffled_feedback_delta",
        "false_positive_delta",
        "false_negative_delta",
        "tautology_risk_cleared",
        "information_asymmetry_enforced",
        "inference_substrate",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock hooks for deterministic Exp 3032 evaluation."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    heldout_replay_path: Path | None = None
    exp3019_artifact_path: Path | None = None
    exp3019_table_path: Path | None = None
    exp3020_artifact_path: Path | None = None
    controller_config_path: Path | None = None
    replay_transcript_path: Path | None = None
    exp3017_manifest_path: Path | None = None
    exp3018_manifest_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_heldout_replay_path(self) -> Path:
        return self.heldout_replay_path or self.repo_root / HELDOUT_REPLAY_REL_PATH

    def resolved_exp3019_artifact_path(self) -> Path:
        return self.exp3019_artifact_path or self.repo_root / EXP3019_ARTIFACT_REL_PATH

    def resolved_exp3019_table_path(self) -> Path:
        return self.exp3019_table_path or self.repo_root / EXP3019_TABLE_REL_PATH

    def resolved_exp3020_artifact_path(self) -> Path:
        return self.exp3020_artifact_path or self.repo_root / EXP3020_ARTIFACT_REL_PATH

    def resolved_controller_config_path(self) -> Path:
        return self.controller_config_path or self.repo_root / CONTROLLER_CONFIG_REL_PATH

    def resolved_replay_transcript_path(self) -> Path:
        return self.replay_transcript_path or self.repo_root / REPLAY_TRANSCRIPT_REL_PATH

    def resolved_exp3017_manifest_path(self) -> Path:
        return self.exp3017_manifest_path or self.repo_root / EXP3017_MANIFEST_REL_PATH

    def resolved_exp3018_manifest_path(self) -> Path:
        return self.exp3018_manifest_path or self.repo_root / EXP3018_MANIFEST_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded prior artifacts and exact rows used by the replay."""

    exp3019_artifact: JsonDict
    exp3019_rows: tuple[JsonDict, ...]
    exp3020_artifact: JsonDict
    controller_config: JsonDict
    replay_transcript_rows: tuple[JsonDict, ...]
    exp3017_rows: tuple[JsonDict, ...]
    exp3018_rows: tuple[JsonDict, ...]
    controller_update_row_ids: frozenset[str]


@dataclass(frozen=True)
class HeldoutTrace:
    """Checker-side view of one exact held-out claim.

    This intentionally excludes the candidate role, certificate status,
    held-out label field name, and controller update decision.  The checker gets
    only the exact claim ID, non-label controller features, exact authorities,
    and the expected feasible/infeasible label needed for evaluation.
    """

    trace_id: str
    row_id: str
    item_id: str
    exact_claim_id: str
    expected_label: bool
    expected_authorities: tuple[str, ...]
    checker_features: tuple[str, ...]


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3032 terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        artifact = _blocked_artifact(active, _round(active.clock() - started), blocker)
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    heldout_traces = build_heldout_traces(sources)
    report = evaluate_heldout_replay(heldout_traces, sources.controller_config)
    asymmetry = information_asymmetry_enforced(heldout_traces, sources)
    tautology_exposure = dict(report["tautology_exposure"])
    tautology_exposure["information_asymmetry_enforced"] = asymmetry
    tautology_risk_cleared = bool(tautology_exposure["risk_cleared"] and asymmetry)
    substrate = inference_substrate()
    ready = bool(
        heldout_traces
        and report["feasible_infeasible_auc_delta"] > 0.0
        and report["shuffled_feedback_delta"] <= 0.0
        and report["withheld_authority_delta"] <= 0.0
        and report["false_positive_delta"] <= 0.0
        and report["false_negative_delta"] <= 0.0
        and tautology_risk_cleared
        and asymmetry
        and substrate["live_llm_inference"] is False
        and substrate["model_weight_training"] is False
    )

    replay_rows = [_trace_row(trace, report) for trace in heldout_traces]
    _write_jsonl(active.resolved_heldout_replay_path(), replay_rows)
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_heldout_replay_ready": ready,
        "continuous_self_learning_tested": True,
        "heldout_trace_count": len(heldout_traces),
        "feasible_infeasible_auc_delta": report["feasible_infeasible_auc_delta"],
        "shuffled_feedback_delta": report["shuffled_feedback_delta"],
        "false_positive_delta": report["false_positive_delta"],
        "false_negative_delta": report["false_negative_delta"],
        "tautology_risk_cleared": tautology_risk_cleared,
        "information_asymmetry_enforced": asymmetry,
        "inference_substrate": substrate,
        "honest_verdict": (
            "complete_fr11_heldout_replay_ready"
            if ready
            else "complete_bounded_heldout_replay_not_promoted"
        ),
        "heldout_replay_path": str(
            _relative_to(active.repo_root, active.resolved_heldout_replay_path())
        ),
        "duration_s": _round(active.clock() - started),
        "auc_scores": {
            "baseline_auc": report["baseline_auc"],
            "controller_auc": report["controller_auc"],
            "shuffled_feedback_auc": report["shuffled_feedback_auc"],
            "withheld_authority_auc": report["withheld_authority_auc"],
        },
        "false_rate_report": {
            "baseline_false_positive_rate": report["baseline_false_positive_rate"],
            "controller_false_positive_rate": report["controller_false_positive_rate"],
            "baseline_false_negative_rate": report["baseline_false_negative_rate"],
            "controller_false_negative_rate": report["controller_false_negative_rate"],
        },
        "control_report": {
            "shuffled_feedback_delta": report["shuffled_feedback_delta"],
            "withheld_authority_delta": report["withheld_authority_delta"],
            "negative_controls_rejected": bool(
                report["shuffled_feedback_delta"] <= 0.0
                and report["withheld_authority_delta"] <= 0.0
            ),
        },
        "tautology_exposure": tautology_exposure,
        "controller_update_row_ids": sorted(sources.controller_update_row_ids),
        "source_artifacts": {
            "exp3019_tautology_risk_observed": sources.exp3019_artifact.get(
                "tautology_risk_flag"
            )
            is True,
            "exp3020_controller_ready": sources.exp3020_artifact.get(
                "verifier_feedback_controller_ready"
            )
            is True,
            "controller_weight_count": len(_controller_weights(sources.controller_config)),
        },
        "field_principles": field_principles(),
        "tests_run": list(active.tests_run),
    }
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load prior artifacts and merge persisted controller weights."""

    exp3020_artifact = _read_json(config.resolved_exp3020_artifact_path())
    controller_config = _read_json(config.resolved_controller_config_path())
    if "final_weights" not in controller_config:
        summary = _mapping(exp3020_artifact.get("controller_summary"))
        controller_config["final_weights"] = dict(_mapping(summary.get("final_weights")))
    replay_rows = tuple(_read_jsonl(config.resolved_replay_transcript_path()))
    controller_config["feedback_history"] = [dict(row) for row in replay_rows]
    update_ids = frozenset(
        str(row.get("row_id"))
        for row in replay_rows
        if row.get("row_id") is not None
    )
    return SourceBundle(
        exp3019_artifact=_read_json(config.resolved_exp3019_artifact_path()),
        exp3019_rows=tuple(_read_jsonl(config.resolved_exp3019_table_path())),
        exp3020_artifact=exp3020_artifact,
        controller_config=controller_config,
        replay_transcript_rows=replay_rows,
        exp3017_rows=tuple(_read_jsonl(config.resolved_exp3017_manifest_path())),
        exp3018_rows=tuple(_read_jsonl(config.resolved_exp3018_manifest_path())),
        controller_update_row_ids=update_ids,
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first source-evidence blocker, if any."""

    if not sources.exp3019_artifact:
        return "exp3019_artifact_missing_or_empty"
    if sources.exp3019_artifact.get("_malformed") is True:
        return "exp3019_artifact_malformed"
    if sources.exp3019_artifact.get("feasibility_channel_diagnostic_ready") is not True:
        return "exp3019_not_terminal_ready"
    if not sources.exp3019_rows:
        return "exp3019_diagnostic_table_missing"
    if not sources.exp3020_artifact:
        return "exp3020_artifact_missing_or_empty"
    if sources.exp3020_artifact.get("_malformed") is True:
        return "exp3020_artifact_malformed"
    if sources.exp3020_artifact.get("verifier_feedback_controller_ready") is not True:
        return "exp3020_controller_not_ready"
    if not _controller_weights(sources.controller_config):
        return "exp3020_controller_weights_missing"
    if not sources.replay_transcript_rows:
        return "exp3020_replay_transcript_missing"
    if not sources.exp3017_rows:
        return "exp3017_validator_manifest_missing"
    if not sources.exp3018_rows:
        return "exp3018_certificate_manifest_missing"
    return None


def build_heldout_traces(sources: SourceBundle) -> list[HeldoutTrace]:
    """Construct exact held-out checker traces not used by controller updates."""

    validators = {
        str(row.get("item_id")): row
        for row in sources.exp3017_rows
        if row.get("item_id") is not None
    }
    certificates = {
        str(row.get("row_id")): row
        for row in sources.exp3018_rows
        if row.get("row_id") is not None
    }
    traces: list[HeldoutTrace] = []
    for row in sources.exp3019_rows:
        if row.get("heldout_partition") is not True:
            continue
        if row.get("row_type") != "candidate_frontier":
            continue
        row_id = str(row.get("row_id") or "")
        if row_id in sources.controller_update_row_ids:
            continue
        label = _label_from_feasibility_class(str(row.get("feasibility_class") or ""))
        if label is None:
            continue
        item_id = str(row.get("item_id") or "")
        validator = validators.get(item_id, {})
        certificate = certificates.get(row_id, {})
        authorities = _expected_authorities(validator)
        features = _checker_features(certificate, validator)
        traces.append(
            HeldoutTrace(
                trace_id=f"heldout::{row_id}",
                row_id=row_id,
                item_id=item_id,
                exact_claim_id=str(certificate.get("candidate_sha256") or row.get("source_row_sha256")),
                expected_label=label,
                expected_authorities=tuple(authorities),
                checker_features=tuple(features),
            )
        )
    return traces


def evaluate_heldout_replay(
    traces: Sequence[HeldoutTrace],
    controller_config: Mapping[str, Any],
) -> JsonDict:
    """Score held-out traces and compare real replay against controls."""

    baseline_scores = [0.0 for _ in traces]
    weights = _controller_weights(controller_config)
    controller_scores = [_score(weights, trace.checker_features) for trace in traces]
    shuffled_weights = _shuffled_feedback_weights(controller_config)
    shuffled_scores = [_score(shuffled_weights, trace.checker_features) for trace in traces]
    withheld_scores = [
        _score(weights, _withhold_authority_features(trace.checker_features))
        for trace in traces
    ]
    labels = [trace.expected_label for trace in traces]
    baseline_auc = mann_whitney_auc(
        [score for score, label in zip(baseline_scores, labels, strict=True) if label],
        [score for score, label in zip(baseline_scores, labels, strict=True) if not label],
    )
    controller_auc = mann_whitney_auc(
        [score for score, label in zip(controller_scores, labels, strict=True) if label],
        [score for score, label in zip(controller_scores, labels, strict=True) if not label],
    )
    shuffled_auc = mann_whitney_auc(
        [score for score, label in zip(shuffled_scores, labels, strict=True) if label],
        [score for score, label in zip(shuffled_scores, labels, strict=True) if not label],
    )
    withheld_auc = mann_whitney_auc(
        [score for score, label in zip(withheld_scores, labels, strict=True) if label],
        [score for score, label in zip(withheld_scores, labels, strict=True) if not label],
    )
    baseline_fp, baseline_fn = false_rates(baseline_scores, labels)
    controller_fp, controller_fn = false_rates(controller_scores, labels)
    prohibited_checker = sum(
        1
        for trace in traces
        for feature in trace.checker_features
        if feature in PROHIBITED_CHECKER_FIELDS
    )
    prohibited_weights = sum(1 for feature in weights if feature in PROHIBITED_CHECKER_FIELDS)
    return {
        "baseline_auc": baseline_auc,
        "controller_auc": controller_auc,
        "shuffled_feedback_auc": shuffled_auc,
        "withheld_authority_auc": withheld_auc,
        "feasible_infeasible_auc_delta": _round(controller_auc - baseline_auc),
        "shuffled_feedback_delta": _round(shuffled_auc - baseline_auc),
        "withheld_authority_delta": _round(withheld_auc - baseline_auc),
        "baseline_false_positive_rate": baseline_fp,
        "controller_false_positive_rate": controller_fp,
        "baseline_false_negative_rate": baseline_fn,
        "controller_false_negative_rate": controller_fn,
        "false_positive_delta": _round(controller_fp - baseline_fp),
        "false_negative_delta": _round(controller_fn - baseline_fn),
        "controller_scores": dict(
            zip([trace.row_id for trace in traces], controller_scores, strict=True)
        ),
        "baseline_scores": dict(zip([trace.row_id for trace in traces], baseline_scores, strict=True)),
        "tautology_exposure": {
            "source_tautology_flag_observed": True,
            "training_row_overlap_count": 0,
            "prohibited_checker_field_count": prohibited_checker,
            "prohibited_controller_weight_count": prohibited_weights,
            "shuffled_feedback_improved": shuffled_auc > baseline_auc,
            "withheld_authority_improved": withheld_auc > baseline_auc,
            "risk_cleared": bool(
                traces
                and prohibited_checker == 0
                and prohibited_weights == 0
                and shuffled_auc <= baseline_auc
                and withheld_auc <= baseline_auc
            ),
        },
    }


def information_asymmetry_enforced(
    traces: Sequence[HeldoutTrace],
    sources: SourceBundle,
) -> bool:
    """Verify checker traces are disjoint from controller updates and labels."""

    if not traces:
        return False
    for trace in traces:
        if trace.row_id in sources.controller_update_row_ids:
            return False
        if not trace.expected_authorities:
            return False
        if PROHIBITED_CHECKER_FIELDS & set(trace.checker_features):
            return False
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the terminal artifact violates the Exp 3032 contract."""

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

    ready = artifact.get("fr11_heldout_replay_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if ready:
        if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must use a terminal completion prefix")
        if artifact.get("continuous_self_learning_tested") is not True:
            raise ValueError("continuous_self_learning_tested must be true")
        if int(artifact.get("heldout_trace_count") or 0) <= 0:
            raise ValueError("heldout_trace_count must be positive")
        if float(artifact.get("feasible_infeasible_auc_delta") or 0.0) <= 0.0:
            raise ValueError("feasible_infeasible_auc_delta must be positive")
        if float(artifact.get("shuffled_feedback_delta") or 0.0) > 0.0:
            raise ValueError("shuffled_feedback_delta must not improve")
        if float(artifact.get("false_positive_delta") or 0.0) > 0.0:
            raise ValueError("false_positive_delta must not increase")
        if float(artifact.get("false_negative_delta") or 0.0) > 0.0:
            raise ValueError("false_negative_delta must not increase")
        if artifact.get("tautology_risk_cleared") is not True:
            raise ValueError("tautology_risk_cleared must be true")
        if artifact.get("information_asymmetry_enforced") is not True:
            raise ValueError("information_asymmetry_enforced must be true")
    elif verdict.startswith(BLOCKED_PREFIXES):
        if int(artifact.get("heldout_trace_count") or 0) != 0:
            raise ValueError("blocked artifacts must have zero heldout_trace_count")
    elif not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use complete or blocked prefix")


def mann_whitney_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Compute AUROC from pairwise positive/negative score ordering."""

    if not positive_scores or not negative_scores:
        return 0.0
    wins = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return _round(wins / (len(positive_scores) * len(negative_scores)))


def false_rates(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    """Return false-positive and false-negative rates at the zero threshold."""

    negatives = [score for score, label in zip(scores, labels, strict=True) if not label]
    positives = [score for score, label in zip(scores, labels, strict=True) if label]
    false_positive = (
        sum(1 for score in negatives if score >= 0.0) / len(negatives) if negatives else 0.0
    )
    false_negative = (
        sum(1 for score in positives if score < 0.0) / len(positives) if positives else 0.0
    )
    return _round(false_positive), _round(false_negative)


def inference_substrate() -> JsonDict:
    """Describe exactly what execution substrate this cached replay used."""

    return {
        "mode": "cached_exact_trace_replay",
        "live_llm_inference": False,
        "model_weight_training": False,
        "controller_weight_training": False,
        "cached_artifacts_only": True,
    }


def field_principles() -> JsonDict:
    """Return compact reasons for required terminal fields."""

    return {
        "fr11_heldout_replay_ready": "Downstream nonforgetting stress gates on held-out replay.",
        "continuous_self_learning_tested": "Milestone must explicitly cover FR-11 self-learning.",
        "heldout_trace_count": "Sample size must be explicit.",
        "feasible_infeasible_auc_delta": "Self-learning must improve exact separation.",
        "shuffled_feedback_delta": "Negative control must reject tautological gains.",
        "false_positive_delta": "Verifier feedback must not increase unsafe acceptance.",
        "false_negative_delta": "Verifier feedback must not suppress valid cases.",
        "tautology_risk_cleared": "FR-11 promotion requires non-vacuous evidence.",
        "information_asymmetry_enforced": "Checker must not grade its own generation.",
        "inference_substrate": "Cached feedback experiments must not claim live LLM inference.",
        "honest_verdict": "Terminal verdict must be machine-readable.",
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    """CLI entrypoint for focused Exp 3032 runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--heldout-replay", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(output_path=args.output, heldout_replay_path=args.heldout_replay)
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["fr11_heldout_replay_ready"] else 1


def _label_from_feasibility_class(value: str) -> bool | None:
    if value == "feasible":
        return True
    if value == "violating":
        return False
    return None


def _expected_authorities(validator_row: Mapping[str, Any]) -> list[str]:
    nodes = _sequence(_mapping(validator_row.get("validator_tree")).get("nodes"))
    authorities = {
        str(_mapping(node).get("authority"))
        for node in nodes
        if _mapping(node).get("authoritative", True)
    }
    return sorted(authority for authority in authorities if authority and authority != "None")


def _checker_features(
    certificate_row: Mapping[str, Any],
    validator_row: Mapping[str, Any],
) -> list[str]:
    outcome = _mapping(certificate_row.get("deterministic_validator_outcome"))
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
    if _mapping(certificate_row.get("frontier_exploration")).get("bounded") is True:
        features.add("frontier::bounded_cached_candidate_set")
    if certificate_row.get("prefix_closed_assumption_applies") is True:
        features.add("frontier::prefix_closed_assumption")
    return sorted(features)


def _validator_nodes_by_id(validator_row: Mapping[str, Any]) -> dict[str, JsonDict]:
    tree = _mapping(validator_row.get("validator_tree"))
    return {
        str(node.get("node_id")): dict(node)
        for node in _sequence(tree.get("nodes"))
        if _mapping(node).get("node_id")
    }


def _controller_weights(controller_config: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(feature): float(weight)
        for feature, weight in _mapping(controller_config.get("final_weights")).items()
    }


def _shuffled_feedback_weights(controller_config: Mapping[str, Any]) -> dict[str, float]:
    rows = _sequence(controller_config.get("feedback_history"))
    if not rows:
        return {}
    labels = [_mapping(row).get("exact_feedback") is True for row in rows]
    shuffled = labels[1:] + labels[:1]
    learning_rate = float(controller_config.get("learning_rate", 0.25))
    max_abs = float(controller_config.get("max_abs_weight", 1.0))
    weights: dict[str, float] = {}
    for row, label in zip(rows, shuffled, strict=True):
        direction = 1.0 if label else -1.0
        for feature in _string_list(_mapping(row).get("features")):
            updated = weights.get(feature, 0.0) + learning_rate * direction
            weights[feature] = max(-max_abs, min(max_abs, updated))
    return weights


def _score(weights: Mapping[str, float], features: Sequence[str]) -> float:
    return _round(sum(float(weights.get(feature, 0.0)) for feature in features))


def _withhold_authority_features(features: Sequence[str]) -> list[str]:
    return [
        feature
        for feature in features
        if not feature.startswith(AUTHORITY_WITHHELD_PREFIXES)
    ]


def _trace_row(trace: HeldoutTrace, report: Mapping[str, Any]) -> JsonDict:
    return {
        "trace_id": trace.trace_id,
        "row_id": trace.row_id,
        "item_id": trace.item_id,
        "exact_claim_id": trace.exact_claim_id,
        "expected_label": trace.expected_label,
        "expected_authorities": list(trace.expected_authorities),
        "checker_features": list(trace.checker_features),
        "baseline_score": _mapping(report.get("baseline_scores")).get(trace.row_id, 0.0),
        "controller_score": _mapping(report.get("controller_scores")).get(trace.row_id, 0.0),
    }


def _blocked_artifact(config: ExperimentConfig, duration_s: float, reason: str) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_heldout_replay_ready": False,
        "continuous_self_learning_tested": True,
        "heldout_trace_count": 0,
        "feasible_infeasible_auc_delta": 0.0,
        "shuffled_feedback_delta": 0.0,
        "false_positive_delta": 0.0,
        "false_negative_delta": 0.0,
        "tautology_risk_cleared": False,
        "information_asymmetry_enforced": False,
        "inference_substrate": inference_substrate(),
        "honest_verdict": f"blocked_{reason}",
        "blocked_reason": reason,
        "heldout_replay_path": str(
            _relative_to(config.repo_root, config.resolved_heldout_replay_path())
        ),
        "duration_s": duration_s,
        "tests_run": list(config.tests_run),
        "field_principles": field_principles(),
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
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value)]


def _safe_token(value: str) -> str:
    cleaned = [char.lower() if char.isalnum() else "_" for char in value]
    return "_".join("".join(cleaned).split("_")).strip("_") or "unknown"


def _round(value: float) -> float:
    return round(float(value), 6)


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


if __name__ == "__main__":  # pragma: no cover - direct module execution.
    raise SystemExit(main())
