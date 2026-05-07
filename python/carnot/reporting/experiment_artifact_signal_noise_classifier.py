"""Classify experiment result artifacts as signal, noise, or ambiguous.

Spec: REQ-REPORT-040, SCENARIO-REPORT-040.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260507"
EXPERIMENT = "1454_experiment_artifact_signal_noise_classifier"
SCHEMA = "experiment_artifact_signal_noise_classifier_v1"
HEURISTIC_VERSION = "exp1454-v1"
OUTPUT_FILE = "experiment_1454_experiment_artifact_signal_noise_classifier.json"

DEFAULT_OUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_TABLE_PATH = REPO_ROOT / "ops" / "experiment_signal_noise_classification.csv"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "ops" / "experiment_signal_noise_summary.md"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "artifacts_scanned",
    "classification_table_path",
    "summary_path",
    "signal_count",
    "noise_count",
    "ambiguous_count",
    "top_50_noise_candidates",
    "heuristic_version",
    "honest_verdict",
}

CSV_COLUMNS = [
    "experiment_id",
    "path",
    "title",
    "status",
    "honest_verdict",
    "headline_fields",
    "gate_fields",
    "retirement_fields",
    "key_metric_fields",
    "classification",
    "reason",
]

HEADLINE_TOKENS = ("headline", "live_gpu", "live_sota", "provenance_ready")
GATE_TOKENS = ("gate", "blocked", "blocker", "missing", "runtime", "preflight")
RETIREMENT_TOKENS = ("retire", "retired", "retirement", "superseded")
METRIC_TOKENS = (
    "accuracy",
    "auroc",
    "auc",
    "delta",
    "effect",
    "improvement",
    "precision",
    "recall",
    "rate",
    "score",
    "count",
    "samples",
    "cases",
)
ENVIRONMENT_TOKENS = (
    "blocked",
    "missing_tool",
    "missing tool",
    "toolchain",
    "cuda",
    "gpu",
    "vram",
    "oom",
    "no_live",
    "runtime",
    "dependency",
    "importerror",
    "install",
    "unavailable",
    "not_cached",
    "cache",
    "missing_artifact",
    "artifact_missing",
    "gate_blocked",
    "gated_missing",
    "upstream",
    "precondition",
    "source_missing",
)
NOISE_TOKENS = (
    "no_improvement",
    "no improvement",
    "negative",
    "regression",
    "below",
    "failed",
    "failure",
    "not_viable",
    "not viable",
    "flat",
    "plateau",
    "zero_growth",
    "no_delta",
    "collapsed",
    "degenerate",
    "worse",
    "no_headline",
)
SIGNAL_TOKENS = (
    "headline",
    "verified",
    "positive",
    "improved",
    "improvement",
    "beats",
    "success",
    "ready",
    "complete",
    "met",
    "passes",
    "closed",
)

_EXPERIMENT_ID_RE = re.compile(r"experiment_(\d+)")


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _relative_path(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        parts = path.parts
        if "results" in parts:
            return str(Path(*parts[parts.index("results") :]))
        if "ops" in parts:
            return str(Path(*parts[parts.index("ops") :]))
        return path.name


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUT_PATH,
    *,
    table_path: Path | str = DEFAULT_TABLE_PATH,
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
    root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """REQ-REPORT-040: seed the run before scanning mutable result artifacts."""

    root_path = Path(root)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-REPORT-040", "SCENARIO-REPORT-040"],
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "in_progress",
        "artifacts_scanned": 0,
        "classification_table_path": _relative_path(Path(table_path), root_path),
        "summary_path": _relative_path(Path(summary_path), root_path),
        "signal_count": 0,
        "noise_count": 0,
        "ambiguous_count": 0,
        "top_50_noise_candidates": [],
        "heuristic_version": HEURISTIC_VERSION,
        "honest_verdict": "in_progress",
    }
    return _write_json(out_path, artifact)


def _read_json_record(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, "malformed_json"
    if not isinstance(payload, dict):
        return {}, "non_object_json"
    return payload, None


def _safe_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        if len(value) <= 5 and all(not isinstance(item, dict | list) for item in value):
            return "[" + ", ".join(_safe_value(item) for item in value) + "]"
        return f"list[{len(value)}]"
    text = str(value).replace("\n", " ").strip()
    if len(text) > 140:
        return text[:137] + "..."
    return text


def _iter_leaf_fields(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if not isinstance(value, Mapping):  # pragma: no cover - callers pass artifact mappings.
        return [(prefix, value)] if prefix else []
    fields: list[tuple[str, Any]] = []
    for key in sorted(value):
        field_name = f"{prefix}.{key}" if prefix else str(key)
        child = value[key]
        if isinstance(child, Mapping):
            fields.extend(_iter_leaf_fields(child, field_name))
        elif isinstance(child, list):
            fields.append((field_name, child))
        else:
            fields.append((field_name, child))
    return fields


def _filtered_fields(payload: Mapping[str, Any], tokens: tuple[str, ...]) -> str:
    matches: list[str] = []
    for key, value in _iter_leaf_fields(payload):
        lowered = key.lower()
        if any(token in lowered for token in tokens):
            matches.append(f"{key}={_safe_value(value)}")
    return "; ".join(matches[:20])


def _key_metric_fields(payload: Mapping[str, Any]) -> str:
    metrics: list[str] = []
    for key, value in _iter_leaf_fields(payload):
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        lowered = key.lower()
        if any(token in lowered for token in METRIC_TOKENS):
            metrics.append(f"{key}={_safe_value(value)}")
    return "; ".join(metrics[:20])


def _experiment_id(path: Path, payload: Mapping[str, Any]) -> str:
    match = _EXPERIMENT_ID_RE.search(path.name)
    if match:
        return match.group(1)
    raw = str(payload.get("experiment", "")).strip()
    return raw.removeprefix("exp") or "unknown"


def _title(path: Path, payload: Mapping[str, Any]) -> str:
    for key in ("title", "experiment_title", "task_title", "name"):
        value = payload.get(key)
        if value:
            return _safe_value(value)
    metadata = payload.get("artifact_metadata")
    if isinstance(metadata, Mapping) and metadata.get("title"):
        return _safe_value(metadata["title"])
    stem = _EXPERIMENT_ID_RE.sub("", path.stem).strip("_")
    return stem.replace("_", " ")


def _decision_text(payload: Mapping[str, Any]) -> str:
    pieces = [
        _safe_value(payload.get("status", "")),
        _safe_value(payload.get("honest_verdict", "")),
        _safe_value(payload.get("verdict", "")),
        _safe_value(payload.get("outcome", "")),
        _safe_value(payload.get("reason", "")),
    ]
    return " ".join(piece for piece in pieces if piece).lower()


def _has_truthy_headline(payload: Mapping[str, Any]) -> bool:
    for key, value in _iter_leaf_fields(payload):
        if "headline" in key.lower() and value is True:
            return True
    return False


def _has_retirement(payload: Mapping[str, Any], decision_text: str) -> bool:
    if str(payload.get("status", "")).lower() == "retired":
        return True
    if "retired" in decision_text or "retire_if_same_verdict" in decision_text:
        return True
    for key in ("retirement_reason", "retired_reason", "retire_reason", "superseded_by"):
        if payload.get(key):
            return True
    return payload.get("retire_if_same_verdict") is True and _has_noise_evidence(decision_text)


def _has_environmental_blocker(text: str) -> bool:
    return any(token in text for token in ENVIRONMENT_TOKENS)


def _has_noise_evidence(text: str) -> bool:
    return any(token in text for token in NOISE_TOKENS)


def _has_signal_evidence(text: str) -> bool:
    return any(token in text for token in SIGNAL_TOKENS)


def _classification_reason(path: Path, payload: Mapping[str, Any], error: str | None) -> tuple[str, str]:
    decision_text = _decision_text(payload)
    gate_text = f"{decision_text} {_filtered_fields(payload, GATE_TOKENS)}".lower()
    status = str(payload.get("status", "")).lower()
    if error:
        return "AMBIGUOUS", f"{error}; artifact is malformed or not a JSON object"
    if path.name == OUTPUT_FILE:
        return "AMBIGUOUS", "classifier self-artifact metadata, not a scientific result"
    if _has_retirement(payload, decision_text):
        return "NOISE", "explicit retirement or retire-if-same-verdict marker"
    if _has_environmental_blocker(gate_text) and status in {"blocked", "missing", "gated", "failed"}:
        return "AMBIGUOUS", "environmental or upstream blocker, not scientific noise"
    if status in {"in_progress", "partial"}:
        return "AMBIGUOUS", "artifact is not terminal"
    if _has_noise_evidence(decision_text):
        return "NOISE", "merit-gate negative, no-improvement, regression, or no-headline evidence"
    if _has_truthy_headline(payload):
        return "SIGNAL", "headline eligibility or headline provenance is explicitly true"
    if status in {"complete", "success", "succeeded"} and _has_signal_evidence(decision_text):
        return "SIGNAL", "completed artifact records positive, verified, or closed evidence"
    if status in {"complete", "success", "succeeded"}:
        return "SIGNAL", "terminal success without an explicit negative or blocker verdict"
    if status == "failed":
        return "NOISE", "terminal failure without an environmental blocker"
    return "AMBIGUOUS", "insufficient transparent evidence for signal or noise"


def classify_artifact(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    error: str | None = None,
    root: Path | str = REPO_ROOT,
) -> dict[str, str]:
    """Return one deterministic CSV row for an experiment artifact."""

    artifact_path = Path(path)
    classification, reason = _classification_reason(artifact_path, payload, error)
    return {
        "experiment_id": _experiment_id(artifact_path, payload),
        "path": _relative_path(artifact_path, Path(root)),
        "title": _title(artifact_path, payload),
        "status": _safe_value(payload.get("status", "unreadable" if error else "")),
        "honest_verdict": _safe_value(payload.get("honest_verdict", "")),
        "headline_fields": _filtered_fields(payload, HEADLINE_TOKENS),
        "gate_fields": _filtered_fields(payload, GATE_TOKENS),
        "retirement_fields": _filtered_fields(payload, RETIREMENT_TOKENS),
        "key_metric_fields": _key_metric_fields(payload),
        "classification": classification,
        "reason": reason,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _candidate(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "experiment_id": row["experiment_id"],
        "path": row["path"],
        "title": row["title"],
        "reason": row["reason"],
    }


def _noise_score(row: Mapping[str, str]) -> tuple[int, str]:
    text = f"{row['honest_verdict']} {row['retirement_fields']}".lower()
    if "explicit retirement" in row["reason"].lower():
        return (0, row["experiment_id"])
    if "no-improvement" in text or "no_improvement" in text:
        return (1, row["experiment_id"])
    if "regression" in text or "negative" in text:
        return (2, row["experiment_id"])
    return (3, row["experiment_id"])


def _write_summary(path: Path, rows: list[dict[str, str]], counts: Mapping[str, int]) -> None:
    noise_rows = sorted(
        (row for row in rows if row["classification"] == "NOISE"),
        key=_noise_score,
    )[:50]
    signal_rows = [row for row in rows if row["classification"] == "SIGNAL"][:25]
    ambiguous_rows = [row for row in rows if row["classification"] == "AMBIGUOUS"][:25]
    lines = [
        "# Experiment Signal / Noise Classification Summary",
        "",
        f"Run date: `{RUN_DATE}`",
        f"Heuristic version: `{HEURISTIC_VERSION}`",
        "",
        "## Counts",
        "",
        f"- SIGNAL: {counts['SIGNAL']}",
        f"- NOISE: {counts['NOISE']}",
        f"- AMBIGUOUS: {counts['AMBIGUOUS']}",
        "",
        "## Top 50 Noise Candidates",
        "",
    ]
    lines.extend(f"- Exp {row['experiment_id']}: {row['title']} - {row['reason']}" for row in noise_rows)
    lines.extend(["", "## Top Signal Candidates", ""])
    lines.extend(f"- Exp {row['experiment_id']}: {row['title']} - {row['reason']}" for row in signal_rows)
    lines.extend(["", "## Ambiguous Operator-Decision Items", ""])
    lines.extend(
        f"- Exp {row['experiment_id']}: {row['title']} - {row['reason']}"
        for row in ambiguous_rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    table_path: Path | str = DEFAULT_TABLE_PATH,
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
) -> dict[str, Any]:
    """Run the full Exp 1454 scan and write CSV, markdown, and JSON artifacts."""

    root_path = Path(root)
    out = Path(out_path)
    table = Path(table_path)
    summary = Path(summary_path)
    write_in_progress_artifact(out, table_path=table, summary_path=summary, root=root_path)
    result_paths = sorted((root_path / "results").glob("experiment_*.json"))
    rows: list[dict[str, str]] = []
    for result_path in result_paths:
        payload, error = _read_json_record(result_path)
        rows.append(classify_artifact(result_path, payload, error=error, root=root_path))
    counts = {
        "SIGNAL": sum(row["classification"] == "SIGNAL" for row in rows),
        "NOISE": sum(row["classification"] == "NOISE" for row in rows),
        "AMBIGUOUS": sum(row["classification"] == "AMBIGUOUS" for row in rows),
    }
    _write_csv(table, rows)
    _write_summary(summary, rows, counts)
    noise_rows = sorted(
        (row for row in rows if row["classification"] == "NOISE"),
        key=_noise_score,
    )[:50]
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec": ["REQ-REPORT-040", "SCENARIO-REPORT-040"],
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "artifacts_scanned": len(rows),
        "classification_table_path": _relative_path(table, root_path),
        "summary_path": _relative_path(summary, root_path),
        "signal_count": counts["SIGNAL"],
        "noise_count": counts["NOISE"],
        "ambiguous_count": counts["AMBIGUOUS"],
        "top_50_noise_candidates": [_candidate(row) for row in noise_rows],
        "heuristic_version": HEURISTIC_VERSION,
        "honest_verdict": (
            f"complete_exp1454_signal_noise_ledger_written_{len(rows)}_artifacts_"
            f"{counts['SIGNAL']}_signal_{counts['NOISE']}_noise_{counts['AMBIGUOUS']}_ambiguous"
        ),
        "classification_table_written": True,
        "summary_written": True,
    }
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
