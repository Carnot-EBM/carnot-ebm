"""Reusable capstone aggregation helper that reports gaps per axis.

Spec refs: REQ-CAPSTONE-4308, SCENARIO-CAPSTONE-4308.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


JsonDict = dict[str, Any]
AxisVerdictFn = Callable[[Mapping[str, JsonDict]], Any]


@dataclass(frozen=True)
class AxisSpec:
    """One independently-computed capstone axis."""

    name: str
    required_keys: tuple[str, ...]
    verdict_fn: AxisVerdictFn


def _experiment_id(
    artifact_key: str,
    artifact_experiment_ids: Mapping[str, int] | None,
) -> int | None:
    if artifact_experiment_ids is None:
        return None
    value = artifact_experiment_ids.get(artifact_key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _missing_gap(
    axis: str,
    artifact_key: str,
    artifact_experiment_ids: Mapping[str, int] | None,
) -> JsonDict:
    gap: JsonDict = {"axis": axis, "artifact_key": artifact_key}
    experiment_id = _experiment_id(artifact_key, artifact_experiment_ids)
    if experiment_id is not None:
        gap["experiment_id"] = experiment_id
    return gap


def _flagged_gap(
    axis: str,
    artifact_key: str,
    artifact_experiment_ids: Mapping[str, int] | None,
) -> JsonDict:
    gap = _missing_gap(axis, artifact_key, artifact_experiment_ids)
    gap["reason"] = "flagged_adversarial"
    return gap


def _present_payload(payload: Any) -> JsonDict | None:
    if not isinstance(payload, Mapping):
        return None
    if payload.get("flagged_adversarial") is True:
        return None
    return dict(payload)


def aggregate_available_report_gaps(
    artifacts: Mapping[str, Any],
    axes: Sequence[AxisSpec],
    *,
    artifact_experiment_ids: Mapping[str, int] | None = None,
) -> JsonDict:
    """Compute each capstone axis from the artifacts available to that axis.

    Missing or flagged artifacts are recorded as gaps for their own axis, but
    they are not allowed to erase verdicts for unrelated axes. The caller owns
    the axis-specific verdict function so capstones can keep their domain logic
    near the capstone while sharing this availability discipline.
    """
    axis_reports: dict[str, JsonDict] = {}
    all_missing: list[JsonDict] = []
    all_flagged: list[JsonDict] = []
    all_available: set[str] = set()

    for axis in axes:
        available: dict[str, JsonDict] = {}
        missing: list[JsonDict] = []
        flagged: list[JsonDict] = []

        for artifact_key in axis.required_keys:
            payload = artifacts.get(artifact_key)
            if isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True:
                gap = _flagged_gap(axis.name, artifact_key, artifact_experiment_ids)
                flagged.append(gap)
                all_flagged.append(gap)
                continue
            present = _present_payload(payload)
            if present is None:
                gap = _missing_gap(axis.name, artifact_key, artifact_experiment_ids)
                missing.append(gap)
                all_missing.append(gap)
                continue
            available[artifact_key] = present
            all_available.add(artifact_key)

        try:
            verdict = axis.verdict_fn(available)
            verdict_error = None
        except (KeyError, TypeError, ValueError) as exc:
            verdict = False
            verdict_error = str(exc)

        axis_report: JsonDict = {
            "verdict": verdict,
            "available_artifact_keys": sorted(available),
            "missing_artifacts": missing,
            "flagged_artifacts": flagged,
        }
        if verdict_error is not None:
            axis_report["verdict_error"] = verdict_error
        axis_reports[axis.name] = axis_report

    return {
        "axes": axis_reports,
        "available_artifact_keys": sorted(all_available),
        "missing_upstream_artifacts": all_missing,
        "flagged_artifacts_excluded": all_flagged,
    }
