"""Exp 2953 threshold policy for code-verifier repair filtering.

Exp 2940 proved that the code verifier carries useful information, but AUPRC
is not an operating policy. This module converts the checked-in precision-
recall summary into explicit thresholds for code repair triage. It does not
rerun generation, does not invent missing labels, and does not treat verifier
approval as a standalone correctness proof.

Spec: REQ-CODE-2953, SCENARIO-CODE-2953,
SCENARIO-CODE-2953-BLOCKED.
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
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2953_code_verifier_threshold_policy_v1.json"
ARTIFACT = "experiment_2953_code_verifier_threshold_policy_v1"
SCHEMA = "carnot.code_verifier_threshold_policy.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP2940_REL_PATH = Path("results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json")
EXP2943_REL_PATH = Path("results/experiment_2943_cross_corpus_matrix_v11.json")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "threshold_policy_ready",
    "source_artifacts",
    "operating_points",
    "selected_default_threshold",
    "expected_ppv_at_default",
    "expected_recall_at_default",
    "expected_false_accept_rate_at_default",
    "deployment_boundary",
    "missing_score_distribution",
    "inference_substrate",
    "duration_s",
)

EXP2940_REQUIRED_FIELDS = (
    "precision_recall_curve",
    "code_corpus_candidate_count",
    "code_corpus_positive_count",
    "code_status_energy_values",
)

NEXT_EXP2940_COMMAND = (
    "PYTHONPATH=python .venv/bin/python -m "
    "carnot.reporting.verifier_ensemble_auprc_code_corpora_2940"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the deterministic Exp 2953 artifact builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp2940_path: Path = EXP2940_REL_PATH
    exp2943_path: Path = EXP2943_REL_PATH
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


@dataclass(frozen=True)
class CurvePoint:
    """Measured precision-recall row with false accepts inferred from counts."""

    threshold: float
    ppv: float
    recall: float
    f1: float
    expected_true_accept_count: float
    expected_false_accept_count: float
    expected_false_accept_rate: float


@dataclass(frozen=True)
class PolicyPoint:
    """A named threshold and the deployment boundary attached to it."""

    policy_name: str
    threshold: float
    expected_ppv: float
    expected_recall: float
    expected_false_accept_rate: float
    false_accept_limit: float
    recommended_use: str
    approval_rule: str
    expected_true_accept_count: float
    expected_false_accept_count: float

    def as_dict(self) -> JsonDict:
        return {
            "policy_name": self.policy_name,
            "threshold": self.threshold,
            "expected_ppv": self.expected_ppv,
            "expected_recall": self.expected_recall,
            "expected_false_accept_rate": self.expected_false_accept_rate,
            "false_accept_limit": self.false_accept_limit,
            "recommended_use": self.recommended_use,
            "approval_rule": self.approval_rule,
            "expected_true_accept_count": self.expected_true_accept_count,
            "expected_false_accept_count": self.expected_false_accept_count,
        }


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the threshold policy from checked-in Exp 2940/2943 summaries."""

    config = config or ExperimentConfig()
    started = config.start_time()
    source_artifacts = _source_artifacts(config)
    missing_sources = [
        source["experiment_id"] for source in source_artifacts if source["required"] and not source["present"]
    ]
    if missing_sources:
        return _blocked_artifact(
            config=config,
            started=started,
            source_artifacts=source_artifacts,
            verdict="blocked_upstream_artifact_missing",
            missing_fields=[f"source:{source}" for source in missing_sources],
            missing_score_distribution=False,
        )

    exp2940 = _read_json(_repo_path(config.repo_root, config.exp2940_path))
    exp2943 = _read_json(_repo_path(config.repo_root, config.exp2943_path))
    missing_fields = _missing_exp2940_fields(exp2940)
    if missing_fields:
        return _blocked_artifact(
            config=config,
            started=started,
            source_artifacts=source_artifacts,
            verdict=(
                "blocked_missing_exp2940_score_distribution"
                if "code_status_energy_values" in missing_fields
                else "blocked_missing_exp2940_threshold_summary"
            ),
            missing_fields=missing_fields,
            missing_score_distribution="code_status_energy_values" in missing_fields,
        )

    candidate_count = int(exp2940["code_corpus_candidate_count"])
    positive_count = int(exp2940["code_corpus_positive_count"])
    baseline_auprc = _baseline_auprc(exp2940)
    curve_points = _curve_points(exp2940["precision_recall_curve"], candidate_count, positive_count)
    operating_points = _select_operating_points(curve_points, baseline_auprc)
    default = operating_points[0]

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": (
            "complete: threshold policy ready; "
            f"default_threshold={default.threshold:.4f}; "
            f"ppv={default.expected_ppv:.6f}; "
            f"recall={default.expected_recall:.6f}; "
            f"false_accept_rate={default.expected_false_accept_rate:.6f}"
        ),
        "threshold_policy_ready": True,
        "source_artifacts": source_artifacts,
        "operating_points": [point.as_dict() for point in operating_points],
        "selected_default_threshold": default.threshold,
        "expected_ppv_at_default": default.expected_ppv,
        "expected_recall_at_default": default.expected_recall,
        "expected_false_accept_rate_at_default": default.expected_false_accept_rate,
        "deployment_boundary": _deployment_boundary(default),
        "missing_score_distribution": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "source_summary": _source_summary(exp2940, exp2943, baseline_auprc),
        "missing_fields": [],
        "next_command": None,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist the Exp 2953 artifact under ``results/``."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _source_artifacts(config: ExperimentConfig) -> list[JsonDict]:
    return [
        _source_artifact(
            config.repo_root,
            config.exp2940_path,
            "exp2940",
            "code_corpus_precision_recall_thresholds",
            (
                "code_corpus_auprc",
                "precision_recall_curve",
                "code_corpus_candidate_count",
                "code_corpus_positive_count",
                "code_status_energy_values",
                "paper_v6_recommendation",
            ),
        ),
        _source_artifact(
            config.repo_root,
            config.exp2943_path,
            "exp2943",
            "cross_corpus_deployment_boundary",
            (
                "matrix_v11_ready",
                "per_corpus_auprc.code_corpora",
                "rows_clean",
                "rows_flagged",
            ),
        ),
    ]


def _source_artifact(
    repo_root: Path,
    rel_path: Path,
    experiment_id: str,
    role: str,
    fields_imported: Sequence[str],
) -> JsonDict:
    path = _repo_path(repo_root, rel_path)
    present = path.is_file()
    return {
        "experiment_id": experiment_id,
        "path": str(rel_path),
        "role": role,
        "required": True,
        "present": present,
        "fields_imported": list(fields_imported) if present else [],
        "sha256": _sha256(path) if present else None,
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    started: float,
    source_artifacts: list[JsonDict],
    verdict: str,
    missing_fields: list[str],
    missing_score_distribution: bool,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "threshold_policy_ready": False,
        "source_artifacts": source_artifacts,
        "operating_points": [],
        "selected_default_threshold": None,
        "expected_ppv_at_default": None,
        "expected_recall_at_default": None,
        "expected_false_accept_rate_at_default": None,
        "deployment_boundary": (
            "No deployment boundary is active because the threshold policy is partial."
        ),
        "missing_score_distribution": missing_score_distribution,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "missing_fields": missing_fields,
        "next_command": NEXT_EXP2940_COMMAND if missing_score_distribution else None,
        "tests_run": list(config.tests_run),
        "required_artifact_fields": list(REQUIRED_ARTIFACT_FIELDS),
    }


def _missing_exp2940_fields(exp2940: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in EXP2940_REQUIRED_FIELDS:
        value = exp2940.get(field_name)
        if value is None or (field_name in {"precision_recall_curve", "code_status_energy_values"} and not value):
            missing.append(field_name)
    return missing


def _curve_points(
    curve_rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
    positive_count: int,
) -> list[CurvePoint]:
    negative_count = candidate_count - positive_count
    if candidate_count <= 0 or positive_count <= 0 or negative_count <= 0:
        raise ValueError("Exp 2940 threshold policy requires both positive and negative candidates")

    points: list[CurvePoint] = []
    for row in curve_rows:
        ppv = _number(row.get("ppv"))
        recall = _number(row.get("recall"))
        threshold = _number(row.get("threshold"))
        true_accepts = positive_count * recall
        accepted = true_accepts / ppv if ppv else 0.0
        false_accepts = max(0.0, accepted - true_accepts)
        points.append(
            CurvePoint(
                threshold=threshold,
                ppv=ppv,
                recall=recall,
                f1=_number(row.get("f1")),
                expected_true_accept_count=_rounded(true_accepts),
                expected_false_accept_count=_rounded(false_accepts),
                expected_false_accept_rate=_rounded(false_accepts / negative_count),
            )
        )
    return sorted(points, key=lambda point: point.threshold, reverse=True)


def _select_operating_points(points: Sequence[CurvePoint], baseline_auprc: float) -> list[PolicyPoint]:
    conservative_candidates = [
        point for point in points if point.ppv >= 0.80 and point.expected_false_accept_rate <= 0.02
    ]
    conservative = max(conservative_candidates or list(points), key=lambda point: point.threshold)

    balanced_candidates = [
        point
        for point in points
        if point.threshold < conservative.threshold
        and point.ppv >= 0.70
        and point.expected_false_accept_rate <= 0.05
    ]
    balanced = max(balanced_candidates or [conservative], key=lambda point: (point.recall, -point.threshold))

    permissive_candidates = [
        point
        for point in points
        if point.threshold < balanced.threshold and point.ppv >= 2.0 * baseline_auprc
    ]
    permissive = min(permissive_candidates or [balanced], key=lambda point: point.threshold)

    return [
        _policy_point(
            "conservative",
            conservative,
            false_accept_limit=0.02,
            recommended_use="automated_candidate_filtering",
            approval_rule="Accept for downstream repair only when approval_score >= threshold.",
        ),
        _policy_point(
            "balanced",
            balanced,
            false_accept_limit=0.05,
            recommended_use="repair_queue_triage",
            approval_rule="Prioritize for repair review; keep tests and sandbox gates mandatory.",
        ),
        _policy_point(
            "permissive",
            permissive,
            false_accept_limit=0.50,
            recommended_use="diagnostic_review_only",
            approval_rule="Use only to inspect failure modes; do not auto-accept repairs.",
        ),
    ]


def _policy_point(
    policy_name: str,
    point: CurvePoint,
    *,
    false_accept_limit: float,
    recommended_use: str,
    approval_rule: str,
) -> PolicyPoint:
    return PolicyPoint(
        policy_name=policy_name,
        threshold=point.threshold,
        expected_ppv=point.ppv,
        expected_recall=point.recall,
        expected_false_accept_rate=point.expected_false_accept_rate,
        false_accept_limit=false_accept_limit,
        recommended_use=recommended_use,
        approval_rule=approval_rule,
        expected_true_accept_count=point.expected_true_accept_count,
        expected_false_accept_count=point.expected_false_accept_count,
    )


def _deployment_boundary(default: PolicyPoint) -> str:
    return (
        f"Default threshold {default.threshold:.4f} is scoped to deterministic "
        "candidate filtering and repair triage for Exp2910-like Python code "
        "generation rows with sandbox/test evidence. It is not a standalone "
        "correctness oracle, not a pass-rate claim, and not a deployment accept "
        "gate for non-code tasks, non-Python code, missing extraction, unseen "
        "corpora, or candidates that have not passed parser, static, sandbox, "
        "and available task-test checks."
    )


def _source_summary(
    exp2940: Mapping[str, Any],
    exp2943: Mapping[str, Any],
    baseline_auprc: float,
) -> JsonDict:
    return {
        "code_corpus_auprc": _number(exp2940.get("code_corpus_auprc")),
        "baseline_random_auprc": baseline_auprc,
        "code_corpus_candidate_count": int(exp2940["code_corpus_candidate_count"]),
        "code_corpus_positive_count": int(exp2940["code_corpus_positive_count"]),
        "score_distribution_count": len(exp2940["code_status_energy_values"]),
        "exp2943_matrix_v11_ready": exp2943.get("matrix_v11_ready"),
        "exp2943_code_row_clean": "exp2940_code_corpus_auprc_corrigendum"
        in set(exp2943.get("rows_clean") or []),
    }


def _baseline_auprc(exp2940: Mapping[str, Any]) -> float:
    raw = exp2940.get("code_corpus_baseline_random_auprc")
    if isinstance(raw, Mapping):
        return _number(raw.get("value"))
    return _number(raw)


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _read_json(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"expected numeric threshold-policy value, got {value!r}")
    return float(value)


def _rounded(value: float) -> float:
    return round(value, 12)


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return round(max(0.0, config.clock() - started), 6)


def main() -> int:  # pragma: no cover
    artifact = write_artifact()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
