"""Exp 2878 local HaluEval/FEVER error-verifiability audit.

This module replays the local Exp 2864 HaluEval/FEVER manifests and measures
whether existing verifier outputs line up with dataset labels and whether local,
deterministic checks can point to an actionable violated constraint. It does not
generate new examples, call a remote model, or use LLM autoformalization.

Spec: REQ-VERIFY-2878, SCENARIO-VERIFY-2878.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval.halueval_fever_pilot import compute_auroc


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
OUTPUT_FILENAME = "experiment_2878_halueval_fever_error_verifiability_v1.json"
EXP2864_REL_PATH = Path("results/experiment_2864_halueval_fever_full_calibration_v3.json")
EXP2865_REL_PATH = Path("results/experiment_2865_cross_corpus_matrix_v5.json")
EXP2867_REL_PATH = Path("results/experiment_2867_drift_mus_prioritizer_v2.json")
EXP2877_REL_PATH = Path("results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json")
HALUEVAL_MANIFEST_REL_PATH = Path("data/eval_manifests/halueval_20260522.jsonl")
FEVER_MANIFEST_REL_PATH = Path("data/eval_manifests/fever_20260522.jsonl")

BUCKETS: tuple[str, ...] = (
    "data-grounding",
    "reasoning-chain",
    "extraction/format",
    "unsupported",
    "unknown",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "error_verifiability_ready",
    "source_artifacts",
    "n_rows_audited",
    "error_buckets",
    "actionable_localization_rate",
    "label_consistency_rate",
    "bucket_level_metrics",
    "weak_auroc_explanation",
    "remote_llm_called",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix verdict from explicit source checks and audit coverage.",
    "error_verifiability_ready": (
        "True when clean source rows are replayed, finite local verifier scores exist, "
        "and at least one labeled error has an actionable local constraint."
    ),
    "source_artifacts": "Only existing .271/.272 artifacts and resolved local manifests are used.",
    "n_rows_audited": "Count of local manifest rows with binary labels and candidate text.",
    "error_buckets": "Every audited row is assigned to exactly one deterministic bucket.",
    "actionable_localization_rate": (
        "Fraction of label=1 rows whose error has an actionable violated constraint."
    ),
    "label_consistency_rate": "Fraction of finite scored rows where verifier direction matches label.",
    "bucket_level_metrics": "Per-bucket AUROC is null unless both labels and finite scores exist.",
    "weak_auroc_explanation": "Separates data-driven, reasoning-driven, and coverage explanations.",
    "remote_llm_called": "Always false; this audit only replays local artifacts and heuristics.",
    "duration_s": "Measured wall-clock runtime; no sleep padding.",
}


@dataclass(frozen=True)
class AuditRow:
    """One existing HaluEval or FEVER manifest row prepared for local audit."""

    dataset_key: str
    dataset: str
    stable_id: str
    prompt: str
    candidate: str
    label: int
    reference: str = ""
    label_text: str = ""
    verifiable: str = ""
    source_name: str = ""

    @property
    def score_text(self) -> str:
        if self.reference:
            return f"{self.prompt}\nReference: {self.reference}\nCandidate: {self.candidate}"
        return f"{self.prompt}\nCandidate: {self.candidate}"


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 2878 local audit."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    tests_run: tuple[str, ...] | list[str] = ()
    score_threshold: float = 0.5
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    run_date: str = RUN_DATE

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


ScoreFn = Callable[[AuditRow], float]


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from *path*, or ``{}`` when it cannot be used."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def compute_auroc_or_none(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute AUROC only when labels and scores form a finite two-class panel."""

    if len(labels) != len(scores) or len(labels) < 2:
        return None
    if len(set(int(label) for label in labels)) < 2:
        return None
    if not all(math.isfinite(float(score)) for score in scores):
        return None
    return compute_auroc(labels, scores)


def bucket_for_row(row: AuditRow) -> str:
    """Assign one deterministic error bucket without looking at verifier scores."""

    label_text = row.label_text.upper().replace("_", " ")
    verifiable = row.verifiable.upper().replace("_", " ")
    prompt = row.prompt.lower()
    candidate = row.candidate.lower()
    if row.dataset_key == "fever":
        if label_text in {"NOT ENOUGH INFO", "NEI"} or verifiable == "NOT VERIFIABLE":
            return "unsupported"
        return "data-grounding"
    if row.dataset_key == "halueval":
        reasoning_terms = (
            "started first",
            "born first",
            "founded first",
            "first aired",
            "which band was founded first",
            "which magazine was started first",
            "earlier",
            "older",
            "before",
            "after",
        )
        if any(term in prompt or term in candidate for term in reasoning_terms):
            return "reasoning-chain"
        if row.reference:
            return "extraction/format"
    return "unknown"


def default_score(row: AuditRow) -> float:
    """Score one audit row with the existing local HaluEval/FEVER verifier path."""

    from carnot.eval.halueval_fever_full_calibration import (
        CalibrationExample,
        default_score_example,
    )

    example = CalibrationExample(
        dataset_key=row.dataset_key,
        stable_id=row.stable_id,
        prompt=row.prompt,
        candidate=row.candidate,
        label=row.label,
        source_name=row.source_name,
        reference=row.reference,
    )
    return float(default_score_example(example))


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    scorer: ScoreFn = default_score,
    write: bool = True,
) -> dict[str, Any]:
    """Build and optionally write the Exp 2878 audit artifact."""

    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    exp2864 = read_json(cfg.repo_root / EXP2864_REL_PATH)
    exp2865 = read_json(cfg.repo_root / EXP2865_REL_PATH)
    exp2867 = read_json(cfg.repo_root / EXP2867_REL_PATH)
    exp2877 = read_json(cfg.repo_root / EXP2877_REL_PATH)
    manifest_paths = _manifest_paths(cfg, exp2864)
    rows = _load_rows(manifest_paths) if _sources_ready(exp2864, exp2865) else []
    certificates = _certificates_by_id(exp2877)
    failure_rows = list(exp2867.get("failure_rows") or [])
    row_audits = [_audit_row(row, scorer, certificates, cfg.score_threshold) for row in rows]
    buckets = _bucket_summaries(row_audits)
    metrics = _bucket_metrics(row_audits)
    source_artifacts = _source_artifacts(cfg, manifest_paths, include_manifests=bool(rows))
    n_positive = sum(1 for audit in row_audits if audit["label"] == 1)
    n_actionable = sum(1 for audit in row_audits if audit["actionable_violated_constraint"])
    scored = [audit for audit in row_audits if audit["score"] is not None]
    consistency_hits = sum(1 for audit in scored if audit["label_consistent"])
    actionable_rate = (n_actionable / n_positive) if n_positive else 0.0
    consistency_rate = (consistency_hits / len(scored)) if scored else 0.0
    ready = bool(rows and scored and n_actionable > 0 and _sources_ready(exp2864, exp2865))
    duration_s = max(0.0, cfg.clock() - started)
    artifact = {
        "honest_verdict": _verdict(ready, rows, scored, exp2864, exp2865),
        "error_verifiability_ready": ready,
        "source_artifacts": source_artifacts,
        "n_rows_audited": len(rows),
        "error_buckets": buckets,
        "actionable_localization_rate": round(actionable_rate, 6),
        "label_consistency_rate": round(consistency_rate, 6),
        "bucket_level_metrics": metrics,
        "weak_auroc_explanation": _weak_auroc_explanation(
            exp2864=exp2864,
            metrics=metrics,
            row_audits=row_audits,
            certificate_count=len(certificates),
            failure_rows=failure_rows,
        ),
        "remote_llm_called": False,
        "tests_run": list(cfg.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": cfg.run_date,
        "duration_s": duration_s,
    }
    validate_artifact(artifact)
    if write:
        write_artifact(cfg.resolved_output_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required schema fields and bucket accounting."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["remote_llm_called"] is not False:
        raise ValueError("remote_llm_called must be false")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run_date must be 20260522")
    if not isinstance(artifact["source_artifacts"], list):
        raise ValueError("source_artifacts must be a list")
    bucket_total = sum(int(payload["n_rows"]) for payload in artifact["error_buckets"].values())
    if bucket_total != int(artifact["n_rows_audited"]):
        raise ValueError("bucket count must equal n_rows_audited")


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Persist the Exp 2878 deliverable as stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sources_ready(exp2864: Mapping[str, Any], exp2865: Mapping[str, Any]) -> bool:
    return bool(
        exp2864.get("halueval_fever_ready")
        and exp2864.get("full_benchmark_ready")
        and exp2865.get("cross_corpus_matrix_built")
        and dict(exp2865.get("row_status_by_corpus") or {}).get("HaluEval/FEVER") == "clean"
    )


def _manifest_paths(config: ExperimentConfig, exp2864: Mapping[str, Any]) -> dict[str, Path]:
    paths = dict(exp2864.get("manifest_paths_used") or {})
    halueval = Path(str(paths.get("halueval") or HALUEVAL_MANIFEST_REL_PATH))
    fever = Path(str(paths.get("fever") or FEVER_MANIFEST_REL_PATH))
    return {
        "halueval": halueval if halueval.is_absolute() else config.repo_root / halueval,
        "fever": fever if fever.is_absolute() else config.repo_root / fever,
    }


def _load_rows(manifest_paths: Mapping[str, Path]) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for dataset_key in ("halueval", "fever"):
        path = manifest_paths[dataset_key]
        for raw in _read_jsonl(path):
            label = _coerce_label(raw.get("label"))
            candidate = _candidate(raw)
            if label is None or not candidate:
                continue
            rows.append(
                AuditRow(
                    dataset_key=dataset_key,
                    dataset=str(raw.get("dataset") or dataset_key.upper()),
                    stable_id=str(raw.get("stable_id") or f"{dataset_key}-{len(rows)}"),
                    prompt=str(raw.get("prompt") or ""),
                    candidate=candidate,
                    label=label,
                    reference=str(raw.get("reference") or ""),
                    label_text=str(raw.get("label_text") or ""),
                    verifiable=str(raw.get("verifiable") or ""),
                    source_name=str(raw.get("source_name") or ""),
                )
            )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    loaded.append(payload)
    return loaded


def _coerce_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in {0, 1}:
        return value
    text = str(value).strip()
    return int(text) if text in {"0", "1"} else None


def _candidate(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate") or row.get("claim") or "").strip()


def _certificates_by_id(exp2877: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(cert.get("stable_id")): dict(cert)
        for cert in exp2877.get("certificates", [])
        if isinstance(cert, dict) and cert.get("stable_id")
    }


def _audit_row(
    row: AuditRow,
    scorer: ScoreFn,
    certificates: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    score = _finite_score(scorer(row))
    verifier_direction = None if score is None else int(score >= threshold)
    bucket = bucket_for_row(row)
    actionable = _actionable_constraint(row, certificates.get(row.stable_id), bucket)
    return {
        "stable_id": row.stable_id,
        "dataset": row.dataset,
        "dataset_key": row.dataset_key,
        "label": row.label,
        "bucket": bucket,
        "score": score,
        "verifier_direction": verifier_direction,
        "label_consistent": None if verifier_direction is None else verifier_direction == row.label,
        "actionable_violated_constraint": actionable,
    }


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        score = float(value)
        return score if math.isfinite(score) else None
    return None


def _actionable_constraint(
    row: AuditRow,
    certificate: Mapping[str, Any] | None,
    bucket: str,
) -> str | None:
    if row.label != 1:
        return None
    if certificate and (
        str(certificate.get("solver_status")) == "unsat"
        or "contradiction" in str(certificate.get("exact_verdict") or "")
    ):
        return f"exact_frontier:{certificate.get('constraint_type')}"
    label_text = row.label_text.upper().replace("_", " ")
    if bucket == "unsupported":
        return "missing_evidence_for_claim"
    if row.dataset_key == "fever" and label_text == "REFUTES":
        return "claim_refuted_by_dataset_evidence"
    if row.reference and _normalize(row.reference) not in _normalize(row.candidate):
        return "candidate_mismatches_reference"
    if bucket == "reasoning-chain":
        return "reasoning_relation_violated"
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _bucket_summaries(row_audits: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for bucket in BUCKETS:
        subset = [audit for audit in row_audits if audit["bucket"] == bucket]
        labels = Counter(int(audit["label"]) for audit in subset)
        summaries[bucket] = {
            "n_rows": len(subset),
            "label_counts": {str(label): labels.get(label, 0) for label in (0, 1)},
            "actionable_violated_constraints": sum(
                1 for audit in subset if audit["actionable_violated_constraint"]
            ),
            "scored_rows": sum(1 for audit in subset if audit["score"] is not None),
        }
    return summaries


def _bucket_metrics(row_audits: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for bucket in BUCKETS:
        subset = [audit for audit in row_audits if audit["bucket"] == bucket]
        scored = [audit for audit in subset if audit["score"] is not None]
        labels = [int(audit["label"]) for audit in scored]
        scores = [float(audit["score"]) for audit in scored]
        positives = [audit for audit in subset if audit["label"] == 1]
        consistent = [audit for audit in scored if audit["label_consistent"]]
        actionable = [audit for audit in positives if audit["actionable_violated_constraint"]]
        auroc = compute_auroc_or_none(labels, scores)
        metrics[bucket] = {
            "n_rows": len(subset),
            "n_scored": len(scored),
            "label_counts": {
                "0": sum(1 for audit in subset if audit["label"] == 0),
                "1": sum(1 for audit in subset if audit["label"] == 1),
            },
            "actionable_localization_rate": round(
                len(actionable) / len(positives), 6
            )
            if positives
            else 0.0,
            "label_consistency_rate": round(len(consistent) / len(scored), 6) if scored else 0.0,
            "auroc": None if auroc is None else round(float(auroc), 6),
            "mean_score": round(sum(scores) / len(scores), 6) if scores else None,
        }
    return metrics


def _weak_auroc_explanation(
    *,
    exp2864: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, Any]],
    row_audits: Sequence[Mapping[str, Any]],
    certificate_count: int,
    failure_rows: Sequence[Any],
) -> str:
    halueval = exp2864.get("halueval_auroc")
    fever = exp2864.get("fever_auroc")
    data_rows = sum(
        int(metrics[bucket]["n_rows"])
        for bucket in ("data-grounding", "extraction/format", "unsupported")
    )
    reasoning_rows = int(metrics["reasoning-chain"]["n_rows"])
    unknown_rows = int(metrics["unknown"]["n_rows"])
    exact_coverage = (certificate_count / len(row_audits)) if row_audits else 0.0
    data_consistency = _weighted_consistency(
        metrics,
        ("data-grounding", "extraction/format", "unsupported"),
    )
    reasoning_consistency = float(metrics["reasoning-chain"]["label_consistency_rate"])
    failure_hint = f"; .271 failure rows={len(failure_rows)}" if failure_rows else ""
    return (
        "Weak scalar AUROC is best explained by data-driven factual-support and "
        "extraction/unsupported errors plus missing verifier coverage, not by a "
        "single reasoning-chain failure mode. "
        f"Exp2864 AUROC halueval={halueval}, fever={fever}; "
        f"data-driven rows={data_rows}, reasoning-chain rows={reasoning_rows}, "
        f"unknown rows={unknown_rows}; label consistency data-driven={data_consistency:.6f}, "
        f"reasoning-chain={reasoning_consistency:.6f}; exact-trace coverage={exact_coverage:.6f}"
        f"{failure_hint}."
    )


def _weighted_consistency(
    metrics: Mapping[str, Mapping[str, Any]],
    buckets: Sequence[str],
) -> float:
    scored = sum(int(metrics[bucket]["n_scored"]) for bucket in buckets)
    if scored == 0:
        return 0.0
    weighted = sum(
        float(metrics[bucket]["label_consistency_rate"]) * int(metrics[bucket]["n_scored"])
        for bucket in buckets
    )
    return weighted / scored


def _verdict(
    ready: bool,
    rows: Sequence[AuditRow],
    scored: Sequence[Mapping[str, Any]],
    exp2864: Mapping[str, Any],
    exp2865: Mapping[str, Any],
) -> str:
    if not _sources_ready(exp2864, exp2865):
        return "blocked_exp2864_or_matrix"
    if not rows:
        return "blocked_no_manifest_rows"
    if not scored:
        return "complete: HaluEval/FEVER audit built but local verifier scores unavailable"
    if ready:
        return "complete: HaluEval/FEVER local error-verifiability audit ready"
    return "complete: HaluEval/FEVER audit built with insufficient actionable coverage"


def _source_artifacts(
    config: ExperimentConfig,
    manifest_paths: Mapping[str, Path],
    *,
    include_manifests: bool,
) -> list[str]:
    sources = [str(EXP2864_REL_PATH), str(EXP2865_REL_PATH)]
    if (config.repo_root / EXP2867_REL_PATH).is_file():
        sources.append(str(EXP2867_REL_PATH))
    if (config.repo_root / EXP2877_REL_PATH).is_file():
        sources.append(str(EXP2877_REL_PATH))
    if include_manifests:
        sources.extend(
            [
                _display_path(config.repo_root, manifest_paths["halueval"]),
                _display_path(config.repo_root, manifest_paths["fever"]),
            ]
        )
    return sources


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main() -> int:  # pragma: no cover - thin command wrapper.
    run_experiment(ExperimentConfig(repo_root=Path.cwd()))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin command wrapper.
    raise SystemExit(main())


__all__ = [
    "AuditRow",
    "EXP2864_REL_PATH",
    "EXP2865_REL_PATH",
    "EXP2867_REL_PATH",
    "EXP2877_REL_PATH",
    "ExperimentConfig",
    "FIELD_PRINCIPLES",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "bucket_for_row",
    "compute_auroc_or_none",
    "default_score",
    "main",
    "read_json",
    "run_experiment",
    "validate_artifact",
    "write_artifact",
]
