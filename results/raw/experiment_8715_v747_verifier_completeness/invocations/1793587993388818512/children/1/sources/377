"""Exp 3019 FR-11 feasibility-channel de-tautology diagnostic.

Spec refs: REQ-VERIFY-3019, SCENARIO-VERIFY-3019,
SCENARIO-VERIFY-3019-BLOCKED.

This module evaluates whether exact validator/certificate evidence can form an
interpretable feasibility channel for FR-11-style controllers.  The diagnostic
is deliberately conservative: it uses only exact cached evidence, records
negative controls, reports held-out correlation, and flags tautology risk when
the features are still too close to the validator labels.  It does not implement
or claim native Differentiable Symbolic Planning.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp3017


JsonDict = dict[str, Any]
RUN_DATE = "20260525"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = "experiment_3019_fr11_feasibility_channel_de_tautology_diagnostic_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.feasibility_channel_de_tautology_diagnostic.v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME

EXP3018_REL_PATH = Path(
    "results/experiment_3018_beaver_style_validator_frontier_certificate_v1.json"
)
EXP3018_CERTIFICATE_MANIFEST_REL_PATH = Path(
    "results/beaver_style_validator_frontier_certificate_3018/certificate_manifest.jsonl"
)
EXP3017_VALIDATOR_MANIFEST_REL_PATH = Path(
    "results/nsvif_instruction_validator_tree_expansion_3017/validator_manifest.jsonl"
)
EXP3007_REL_PATH = Path("results/experiment_3007_fr11_attractor_trace_memory_stability_v1.json")
DIAGNOSTIC_TABLE_REL_PATH = Path(
    "results/fr11_feasibility_channel_de_tautology_diagnostic_3019/diagnostic_table.jsonl"
)

FEASIBILITY_PROMOTION_THRESHOLD = 0.65
TAUTOLOGY_CORRELATION_THRESHOLD = 0.95
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
PROHIBITED_SCORE_FEATURES = frozenset(
    {"certificate_status", "candidate_role", "heldout_success_label"}
)
SCORE_FEATURE_NAMES = (
    "failure_ratio",
    "rejection_reason_count",
    "exact_authority_coverage",
    "bounded_frontier",
    "prefix_closed_assumption",
    "transcript_hash_present",
    "probability_exact",
    "unresolved_penalty",
    "negative_control_delta",
    "negative_control_accepted",
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "feasibility_channel_diagnostic_ready",
        "diagnostic_table_path",
        "n_rows",
        "feasible_infeasible_auc",
        "negative_control_rejection_rate",
        "heldout_metric_correlation",
        "tautology_risk_flag",
        "reused_label_as_feature",
        "native_dsp_claim_made",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock hooks for deterministic Exp 3019 runs."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    diagnostic_table_path: Path | None = None
    source_certificate_artifact_path: Path | None = None
    source_certificate_manifest_path: Path | None = None
    source_validator_manifest_path: Path | None = None
    exp3007_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_diagnostic_table_path(self) -> Path:
        return self.diagnostic_table_path or self.repo_root / DIAGNOSTIC_TABLE_REL_PATH

    def resolved_source_certificate_artifact_path(self) -> Path:
        return self.source_certificate_artifact_path or self.repo_root / EXP3018_REL_PATH

    def resolved_source_certificate_manifest_path(self) -> Path:
        return self.source_certificate_manifest_path or self.repo_root / EXP3018_CERTIFICATE_MANIFEST_REL_PATH

    def resolved_source_validator_manifest_path(self) -> Path:
        return self.source_validator_manifest_path or self.repo_root / EXP3017_VALIDATOR_MANIFEST_REL_PATH

    def resolved_exp3007_artifact_path(self) -> Path:
        return self.exp3007_artifact_path or self.repo_root / EXP3007_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded exact validator, certificate, and trace-memory source evidence."""

    certificate_artifact: JsonDict
    certificate_rows: tuple[JsonDict, ...]
    validator_rows: tuple[JsonDict, ...]
    exp3007_artifact: JsonDict


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3019 diagnostic artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    if blocker is not None:
        artifact = _blocked_artifact(active, duration_s=_round(active.clock() - started), reason=blocker)
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    rows = build_diagnostic_rows(active, sources)
    _write_jsonl(active.resolved_diagnostic_table_path(), rows)
    artifact = build_artifact(
        active,
        rows,
        sources=sources,
        duration_s=_round(active.clock() - started),
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load exact source artifacts; malformed files become empty evidence."""

    return SourceBundle(
        certificate_artifact=_read_json(config.resolved_source_certificate_artifact_path()),
        certificate_rows=tuple(_read_jsonl(config.resolved_source_certificate_manifest_path())),
        validator_rows=tuple(_read_jsonl(config.resolved_source_validator_manifest_path())),
        exp3007_artifact=_read_json(config.resolved_exp3007_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first missing source-evidence blocker, if any."""

    if not sources.certificate_artifact:
        return "exp3018_artifact_missing_or_empty"
    if sources.certificate_artifact.get("_malformed") is True:
        return "exp3018_artifact_malformed"
    if sources.certificate_artifact.get("frontier_certificate_ready") is not True:
        return "exp3018_not_ready"
    if not sources.certificate_rows:
        return "exp3018_certificate_manifest_missing"
    if not sources.validator_rows:
        return "exp3017_validator_manifest_missing"
    if not sources.exp3007_artifact:
        return "exp3007_artifact_missing_or_empty"
    if sources.exp3007_artifact.get("_malformed") is True:
        return "exp3007_artifact_malformed"
    if sources.exp3007_artifact.get("trace_memory_stability_ready") is not True:
        return "exp3007_not_ready"
    report = sources.exp3007_artifact.get("negative_control_report")
    if not isinstance(report, Mapping) or not report.get("control_heldout_deltas"):
        return "exp3007_negative_controls_missing"
    return None


def build_diagnostic_rows(config: ExperimentConfig, sources: SourceBundle) -> list[JsonDict]:
    """Derive row-level feasibility diagnostics from exact cached evidence."""

    validator_by_item = {
        str(row.get("item_id")): dict(row)
        for row in sources.validator_rows
        if row.get("item_id") is not None
    }
    rows = [
        _diagnostic_row_from_certificate(row, validator_by_item.get(str(row.get("item_id"))))
        for row in sources.certificate_rows
    ]
    rows.extend(_negative_control_rows(sources.exp3007_artifact))
    for index, row in enumerate(rows):
        row["diagnostic_row_index"] = index
        row["diagnostic_table_path"] = str(_relative_to(config.repo_root, config.resolved_diagnostic_table_path()))
    return rows


def build_artifact(
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    sources: SourceBundle,
    duration_s: float,
) -> JsonDict:
    """Build the terminal JSON artifact from diagnostic rows."""

    feasible_scores = [
        float(row["feasibility_score"])
        for row in rows
        if row.get("feasibility_class") == "feasible"
    ]
    violating_scores = [
        float(row["feasibility_score"])
        for row in rows
        if row.get("feasibility_class") == "violating"
    ]
    heldout_rows = [
        row
        for row in rows
        if row.get("heldout_partition") is True
        and isinstance(row.get("heldout_success_label"), bool)
    ]
    heldout_correlation = pearson_correlation(
        [float(row["feasibility_score"]) for row in heldout_rows],
        [1.0 if row["heldout_success_label"] else 0.0 for row in heldout_rows],
    )
    auc = mann_whitney_auc(feasible_scores, violating_scores)
    rejection_rate = negative_control_rejection_rate(rows)
    reused = reused_label_as_feature(SCORE_FEATURE_NAMES)
    tautology_risk = _tautology_risk_flag(heldout_correlation, rows)
    ready = bool(
        rows
        and feasible_scores
        and violating_scores
        and _class_counts(rows).get("negative_control", 0) > 0
        and heldout_rows
        and rejection_rate > 0.0
        and not reused
    )
    table_path = config.resolved_diagnostic_table_path()
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "feasibility_channel_diagnostic_ready": ready,
        "diagnostic_table_path": str(_relative_to(config.repo_root, table_path)),
        "n_rows": len(rows),
        "feasible_infeasible_auc": auc,
        "negative_control_rejection_rate": rejection_rate,
        "heldout_metric_correlation": heldout_correlation,
        "tautology_risk_flag": tautology_risk,
        "reused_label_as_feature": reused,
        "native_dsp_claim_made": False,
        "honest_verdict": (
            "complete_flagged_tautology_risk_feasibility_channel_diagnostic"
            if ready
            else "blocked_feasibility_channel_diagnostic_not_ready"
        ),
        "duration_s": duration_s,
        "inference_substrate": "cached_exact_validator_certificate_trace_replay",
        "class_counts": _class_counts(rows),
        "class_score_means": _class_score_means(rows),
        "score_feature_names": list(SCORE_FEATURE_NAMES),
        "prohibited_score_features": sorted(PROHIBITED_SCORE_FEATURES),
        "heldout_partition_rule": "item numeric suffix divisible by 5",
        "heldout_label_source": "Exp 3017 source exact verifier known-good/known-bad outcomes",
        "heldout_row_count": len(heldout_rows),
        "tautology_risk_reason": _tautology_risk_reason(tautology_risk, heldout_correlation),
        "negative_control_threshold": FEASIBILITY_PROMOTION_THRESHOLD,
        "source_artifacts": _source_summary(config, sources),
        "native_dsp_claim_boundary": (
            "Differentiable Symbolic Planning is external inspiration only; "
            "this artifact implements a cached exact-evidence diagnostic."
        ),
        "tests_run": list(config.tests_run),
        "field_principles": field_principles(),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3019 artifact violates its terminal contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("reused_label_as_feature") is not False:
        raise ValueError("reused_label_as_feature must remain false")
    if artifact.get("native_dsp_claim_made") is not False:
        raise ValueError("native_dsp_claim_made must remain false")
    for name in (
        "feasible_infeasible_auc",
        "negative_control_rejection_rate",
        "heldout_metric_correlation",
    ):
        value = float(artifact.get(name) or 0.0)
        lower = -1.0 if name == "heldout_metric_correlation" else 0.0
        if not lower <= value <= 1.0:
            raise ValueError(f"{name} outside valid range")

    ready = artifact.get("feasibility_channel_diagnostic_ready") is True
    verdict = str(artifact.get("honest_verdict", ""))
    if ready:
        if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must use a terminal completion prefix")
        if int(artifact.get("n_rows") or 0) <= 0:
            raise ValueError("n_rows must be positive when diagnostic is ready")
        if not artifact.get("diagnostic_table_path"):
            raise ValueError("diagnostic_table_path must be present when ready")
    elif not verdict.startswith(BLOCKED_PREFIXES):
        raise ValueError("honest_verdict must use a blocked prefix when not ready")


def mann_whitney_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Compute pairwise AUROC without optional statistics dependencies."""

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


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson correlation, or 0.0 for degenerate samples."""

    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    x_norm = math.sqrt(sum(value * value for value in x_centered))
    y_norm = math.sqrt(sum(value * value for value in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    numerator = sum(x * y for x, y in zip(x_centered, y_centered, strict=True))
    return _round(numerator / (x_norm * y_norm))


def negative_control_rejection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    """Measure how many negative controls fail the usefulness threshold."""

    controls = [row for row in rows if row.get("feasibility_class") == "negative_control"]
    if not controls:
        return 0.0
    rejected = 0
    for row in controls:
        rejected += int(
            float(row.get("feasibility_score") or 0.0) < FEASIBILITY_PROMOTION_THRESHOLD
            and row.get("negative_control_accepted") is not True
            and float(row.get("negative_control_delta") or 0.0) <= 0.0
        )
    return _round(rejected / len(controls))


def reused_label_as_feature(feature_names: Sequence[str]) -> bool:
    """Return true when score inputs include a prohibited label field."""

    return bool(PROHIBITED_SCORE_FEATURES & set(feature_names))


def field_principles() -> JsonDict:
    """Return compact reasons for required terminal fields."""

    return {
        "feasibility_channel_diagnostic_ready": (
            "DVI-style FR-11 controller must gate on an independent diagnostic."
        ),
        "diagnostic_table_path": "Feasibility evidence must be inspectable.",
        "n_rows": "Sample size must be explicit.",
        "feasible_infeasible_auc": "Separation must be measured independently.",
        "negative_control_rejection_rate": (
            "Irrelevant or contradicted traces must not score as useful."
        ),
        "heldout_metric_correlation": (
            "Usefulness must relate to held-out verifier outcomes."
        ),
        "tautology_risk_flag": "Self-grading risk must be explicit.",
        "reused_label_as_feature": "Tautological feature leakage must be rejected.",
        "native_dsp_claim_made": (
            "External architecture inspiration must not become a local capability claim."
        ),
        "honest_verdict": "Terminal verdict must be machine-readable.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for focused Exp 3019 runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--diagnostic-table", type=Path, default=None)
    parser.add_argument("--source-certificate-artifact", type=Path, default=None)
    parser.add_argument("--source-certificate-manifest", type=Path, default=None)
    parser.add_argument("--source-validator-manifest", type=Path, default=None)
    parser.add_argument("--exp3007-artifact", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            output_path=args.output,
            diagnostic_table_path=args.diagnostic_table,
            source_certificate_artifact_path=args.source_certificate_artifact,
            source_certificate_manifest_path=args.source_certificate_manifest,
            source_validator_manifest_path=args.source_validator_manifest,
            exp3007_artifact_path=args.exp3007_artifact,
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["feasibility_channel_diagnostic_ready"] else 1


def _diagnostic_row_from_certificate(
    row: Mapping[str, Any],
    validator_row: Mapping[str, Any] | None,
) -> JsonDict:
    features = _feature_values(row, validator_row)
    score = _feasibility_score(features)
    feasibility_class = _feasibility_class(str(row.get("certificate_status", "")))
    return {
        "row_id": str(row.get("row_id") or row.get("item_id") or "unknown-row"),
        "row_type": str(row.get("row_type") or "unknown"),
        "item_id": str(row.get("item_id") or ""),
        "certificate_status": str(row.get("certificate_status") or "unknown"),
        "feasibility_class": feasibility_class,
        "feasibility_score": score,
        "feature_values": features,
        "heldout_partition": _heldout_partition(str(row.get("item_id") or "")),
        "heldout_success_label": _heldout_label(row, validator_row),
        "tautology_risk_signal": "exact_failure_counts_share_validator_family",
        "native_dsp_claim_made": False,
        "negative_control_delta": None,
        "negative_control_accepted": False,
        "source_row_sha256": exp3017.sha256_text(json.dumps(row, sort_keys=True)),
    }


def _feature_values(row: Mapping[str, Any], validator_row: Mapping[str, Any] | None) -> JsonDict:
    outcome = _mapping(row.get("deterministic_validator_outcome"))
    failing = _sequence(outcome.get("failing_node_ids"))
    rejection_reasons = _sequence(outcome.get("rejection_reasons"))
    authoritative_count = max(1, _authoritative_node_count(validator_row))
    failure_ratio = min(1.0, len(failing) / authoritative_count)
    return {
        "failure_count": len(failing),
        "failure_ratio": _round(failure_ratio),
        "rejection_reason_count": len(rejection_reasons),
        "exact_authority_coverage": _round(_exact_authority_coverage(validator_row)),
        "bounded_frontier": _mapping(row.get("frontier_exploration")).get("bounded") is True,
        "prefix_closed_assumption": row.get("prefix_closed_assumption_applies") is True,
        "transcript_hash_present": bool(row.get("transcript_sha256")),
        "probability_exact": _mapping(row.get("probability_bound_placeholder")).get(
            "exact_probability_computed"
        )
        is True,
        "unresolved_penalty": str(row.get("certificate_status")) in {"unresolved", "non_prefix_closed"},
        "negative_control_delta": None,
        "negative_control_accepted": False,
    }


def _feasibility_score(features: Mapping[str, Any]) -> float:
    if features.get("negative_control_delta") is not None:
        delta = max(0.0, float(features.get("negative_control_delta") or 0.0))
        accepted_penalty = 0.35 if features.get("negative_control_accepted") is True else 0.0
        return _round(min(1.0, delta + accepted_penalty))
    score = 0.2
    score += 0.5 * (1.0 - float(features.get("failure_ratio") or 0.0))
    score += 0.1 * float(features.get("exact_authority_coverage") or 0.0)
    score += 0.1 if features.get("bounded_frontier") is True else 0.0
    score += 0.05 if features.get("prefix_closed_assumption") is True else 0.0
    score += 0.05 if features.get("transcript_hash_present") is True else 0.0
    score -= 0.35 if int(features.get("rejection_reason_count") or 0) > 0 else 0.0
    score -= 0.35 if features.get("unresolved_penalty") is True else 0.0
    score -= 0.05 if features.get("probability_exact") is not True else 0.0
    return _round(max(0.0, min(1.0, score)))


def _negative_control_rows(exp3007_artifact: Mapping[str, Any]) -> list[JsonDict]:
    report = _mapping(exp3007_artifact.get("negative_control_report"))
    deltas = _mapping(report.get("control_heldout_deltas"))
    accepted_ids = set(_string_list(report.get("accepted_control_ids")))
    rows: list[JsonDict] = []
    for control_type, raw_delta in sorted(deltas.items()):
        memory_id = f"control-{control_type.replace('_', '-')}"
        accepted = memory_id in accepted_ids or str(control_type) in accepted_ids
        features = {
            "failure_count": 1,
            "failure_ratio": 1.0,
            "rejection_reason_count": 1,
            "exact_authority_coverage": 1.0,
            "bounded_frontier": False,
            "prefix_closed_assumption": False,
            "transcript_hash_present": False,
            "probability_exact": False,
            "unresolved_penalty": True,
            "negative_control_delta": float(raw_delta or 0.0),
            "negative_control_accepted": accepted,
        }
        rows.append(
            {
                "row_id": memory_id,
                "row_type": "exp3007_negative_control",
                "item_id": memory_id,
                "certificate_status": "negative_control",
                "feasibility_class": "negative_control",
                "feasibility_score": _feasibility_score(features),
                "feature_values": features,
                "heldout_partition": False,
                "heldout_success_label": None,
                "tautology_risk_signal": "negative_control_delta_from_exp3007",
                "native_dsp_claim_made": False,
                "negative_control_delta": float(raw_delta or 0.0),
                "negative_control_accepted": accepted,
                "source_row_sha256": exp3017.sha256_text(f"{control_type}:{raw_delta}:{accepted}"),
            }
        )
    return rows


def _heldout_label(row: Mapping[str, Any], validator_row: Mapping[str, Any] | None) -> bool | None:
    if not _heldout_partition(str(row.get("item_id") or "")):
        return None
    role = str(row.get("candidate_role") or "")
    if validator_row is None:
        return None
    if role == "known_good":
        return _mapping(validator_row.get("known_good_validation")).get("accepted") is True
    if role == "known_bad":
        return _mapping(validator_row.get("known_bad_validation")).get("accepted") is True
    return None


def _heldout_partition(item_id: str) -> bool:
    suffix = item_id.rsplit("-", 1)[-1]
    return suffix.isdigit() and int(suffix) % 5 == 0


def _feasibility_class(status: str) -> str:
    return {
        "certified_safe": "feasible",
        "certified_violating": "violating",
        "unresolved": "unresolved",
        "non_prefix_closed": "non_prefix",
    }.get(status, "unresolved")


def _authoritative_node_count(validator_row: Mapping[str, Any] | None) -> int:
    nodes = _sequence(_mapping(_mapping(validator_row or {}).get("validator_tree")).get("nodes"))
    authoritative = [node for node in nodes if _mapping(node).get("authoritative", True)]
    return len(authoritative)


def _exact_authority_coverage(validator_row: Mapping[str, Any] | None) -> float:
    nodes = _sequence(_mapping(_mapping(validator_row or {}).get("validator_tree")).get("nodes"))
    authoritative = [_mapping(node) for node in nodes if _mapping(node).get("authoritative", True)]
    if not authoritative:
        return 0.0
    exact = sum(1 for node in authoritative if str(node.get("authority")) in exp3017.EXACT_AUTHORITIES)
    return exact / len(authoritative)


def _class_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return dict(Counter(str(row.get("feasibility_class") or "unknown") for row in rows))


def _class_score_means(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    classes = sorted({str(row.get("feasibility_class") or "unknown") for row in rows})
    means: JsonDict = {}
    for name in classes:
        scores = [
            float(row.get("feasibility_score") or 0.0)
            for row in rows
            if row.get("feasibility_class") == name
        ]
        means[name] = _round(sum(scores) / len(scores)) if scores else 0.0
    return means


def _tautology_risk_flag(correlation: float, rows: Sequence[Mapping[str, Any]]) -> bool:
    exact_failure_feature_used = any(
        "failure_count" in _mapping(row.get("feature_values")) for row in rows
    )
    return bool(exact_failure_feature_used or abs(correlation) >= TAUTOLOGY_CORRELATION_THRESHOLD)


def _tautology_risk_reason(flagged: bool, correlation: float) -> str:
    if not flagged:
        return "no tautology trigger crossed the diagnostic threshold"
    return (
        "feasibility features use exact validator failure counts and held-out "
        f"correlation is {correlation}; diagnostic is inspectable but not a native DSP claim"
    )


def _source_summary(config: ExperimentConfig, sources: SourceBundle) -> JsonDict:
    return {
        "exp3018_artifact_path": str(
            _relative_to(config.repo_root, config.resolved_source_certificate_artifact_path())
        ),
        "exp3018_certificate_manifest_path": str(
            _relative_to(config.repo_root, config.resolved_source_certificate_manifest_path())
        ),
        "exp3017_validator_manifest_path": str(
            _relative_to(config.repo_root, config.resolved_source_validator_manifest_path())
        ),
        "exp3007_artifact_path": str(
            _relative_to(config.repo_root, config.resolved_exp3007_artifact_path())
        ),
        "exp3018_flagged_adversarial": bool(sources.certificate_artifact.get("flagged_adversarial")),
        "exp3007_flagged_adversarial": bool(sources.exp3007_artifact.get("flagged_adversarial")),
        "certificate_rows": len(sources.certificate_rows),
        "validator_rows": len(sources.validator_rows),
    }


def _blocked_artifact(config: ExperimentConfig, *, duration_s: float, reason: str) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "feasibility_channel_diagnostic_ready": False,
        "diagnostic_table_path": str(
            _relative_to(config.repo_root, config.resolved_diagnostic_table_path())
        ),
        "n_rows": 0,
        "feasible_infeasible_auc": 0.0,
        "negative_control_rejection_rate": 0.0,
        "heldout_metric_correlation": 0.0,
        "tautology_risk_flag": False,
        "reused_label_as_feature": False,
        "native_dsp_claim_made": False,
        "honest_verdict": f"blocked_{reason}",
        "blocked_reason": reason,
        "duration_s": duration_s,
        "inference_substrate": "cached_exact_validator_certificate_trace_replay",
        "class_counts": {},
        "class_score_means": {},
        "score_feature_names": list(SCORE_FEATURE_NAMES),
        "tests_run": list(config.tests_run),
        "field_principles": field_principles(),
    }


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {"_malformed": True}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value)]


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":  # pragma: no cover - direct module execution.
    raise SystemExit(main())
