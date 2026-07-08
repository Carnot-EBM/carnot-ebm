"""Exp5384: CPU p-bit boundary/overwrite joint diagnostic.

Spec refs: REQ-VERIFY-5384, SCENARIO-VERIFY-5384.

This diagnostic connects two advisory lanes without upgrading either one into
authority. The p-bit boundary lane explains how communication cadence changes
CPU sampler conflict and convergence telemetry. The overwrite lane explains
whether the symbolic solver can still complete, overwrite, reject, or fallback
from hints before accepting an output. Only fixture ids shared by both lanes
are scored, and no hardware speedup is claimed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5371_pbit_boundary_exchange_schedule_v489 as exp5371
from carnot import experiment_5383_overwrite_guidance_scale_validity_v490 as exp5383


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5384_pbit_boundary_overwrite_joint_diagnostic_v490.json"
)
EXPERIMENT = 5384
EXPERIMENT_ID = "exp5384-pbit-boundary-overwrite-joint-diagnostic-v490"
MILESTONE = "2026.07.490"
RUN_DATE = "20260708"
SCHEMA = "carnot.experiment_5384.pbit_boundary_overwrite_joint_diagnostic.v490"
SPEC_REFS = ("REQ-VERIFY-5384", "SCENARIO-VERIFY-5384")
TERMINAL_PREFIXES = ("complete:", "blocked_")

ETA_VALUES = exp5371.ETA_VALUES
HINT_CLASS_NAMES = exp5383.HINT_CLASS_NAMES
EXPECTED_SHARED_FIXTURE_COUNT = 3
COMPARISON_VARIANTS = (
    "monolithic_pbit",
    "boundary_exchange",
    "overwrite_guided",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "complete only if the CPU diagnostic ran or honest_blocked if prerequisites are missing."
    ),
    "pbit_boundary_overwrite_ready": (
        "true only if results are interpretable and unsafe_false_accepts=0."
    ),
    "simulation_only": "must be true.",
    "hardware_speedup_claim": "must be false.",
    "fixture_count": "number of shared fixtures.",
    "eta_values": "list of boundary-exchange ratios tested.",
    "eta_threshold_estimate": (
        "estimated threshold where boundary exchange approaches monolithic behavior."
    ),
    "solver_overwrite_enabled": ("whether overwrite guidance was enabled in the joint condition."),
    "conflict_delta_vs_monolithic": ("conflict count difference vs monolithic p-bit run."),
    "convergence_delta_vs_monolithic": ("convergence step difference vs monolithic p-bit run."),
    "post_projection_validity_rate": "fraction valid after solver projection.",
    "fallback_completeness_rate": "fraction safely completed by fallback.",
    "unsafe_false_accepts": "count of invalid outputs accepted as valid.",
    "honest_verdict": "one-line result or block reason.",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "pbit_boundary_overwrite_ready",
    "simulation_only",
    "hardware_speedup_claim",
    "fixture_count",
    "eta_values",
    "eta_threshold_estimate",
    "solver_overwrite_enabled",
    "conflict_delta_vs_monolithic",
    "convergence_delta_vs_monolithic",
    "post_projection_validity_rate",
    "fallback_completeness_rate",
    "unsafe_false_accepts",
    "honest_verdict",
)


@dataclass(frozen=True)
class SharedFixtureLink:
    """One fixture id present in both boundary and overwrite diagnostics."""

    fixture_id: str
    fixture_class: str
    monolithic_row: JsonDict
    boundary_rows: tuple[JsonDict, ...]
    overwrite_rows: tuple[JsonDict, ...]


def load_joint_sources() -> JsonDict:
    """Load and validate the two upstream artifacts before joining fixtures."""

    boundary_artifact = _load_json(exp5371.RESULT_RELATIVE_PATH)
    overwrite_artifact = _load_json(exp5383.RESULT_RELATIVE_PATH)
    exp5371.validate_artifact(boundary_artifact)
    exp5383.validate_artifact(overwrite_artifact)

    boundary_rows = tuple(boundary_artifact["boundary_exchange_results"])
    overwrite_rows = tuple(overwrite_artifact["matrix_results"])
    boundary_fixture_ids = {str(row["instance_id"]) for row in boundary_rows}
    overwrite_fixture_ids = {
        str(row["source_fixture_id"])
        for row in overwrite_rows
        if row["guidance_mode"] == "overwrite_capable"
    }
    shared_fixture_ids = sorted(boundary_fixture_ids & overwrite_fixture_ids)
    source_artifacts = _unique_strings(
        [
            str(exp5371.RESULT_RELATIVE_PATH),
            str(exp5383.RESULT_RELATIVE_PATH),
            *boundary_artifact["source_artifacts"],
            *overwrite_artifact["source_artifacts"],
        ]
    )
    return {
        "boundary_rows": boundary_rows,
        "overwrite_rows": overwrite_rows,
        "shared_fixture_ids": shared_fixture_ids,
        "source_readiness": {
            "exp5371_boundary_exchange_schedule_ready": bool(
                boundary_artifact["boundary_exchange_schedule_ready"]
            ),
            "exp5383_overwrite_guidance_scale_ready": bool(
                overwrite_artifact["overwrite_guidance_scale_ready"]
            ),
        },
        "source_artifacts": source_artifacts,
    }


def build_shared_fixture_links(
    sources: Mapping[str, Any] | None = None,
) -> tuple[SharedFixtureLink, ...]:
    """Return the shared fixture rows with monolithic, eta, and overwrite data."""

    loaded = load_joint_sources() if sources is None else sources
    links: list[SharedFixtureLink] = []
    for fixture_id in loaded["shared_fixture_ids"]:
        boundary_rows = [
            dict(row) for row in loaded["boundary_rows"] if row["instance_id"] == fixture_id
        ]
        monolithic_row = next(
            row for row in boundary_rows if row["exchange_mode"] == "monolithic_baseline"
        )
        eta_rows = tuple(
            sorted(
                (row for row in boundary_rows if row["eta"] is not None),
                key=lambda row: ETA_VALUES.index(float(row["eta"])),
            )
        )
        overwrite_rows = tuple(
            sorted(
                (
                    dict(row)
                    for row in loaded["overwrite_rows"]
                    if row["source_fixture_id"] == fixture_id
                    and row["guidance_mode"] == "overwrite_capable"
                ),
                key=lambda row: HINT_CLASS_NAMES.index(str(row["hint_class"])),
            )
        )
        links.append(
            SharedFixtureLink(
                fixture_id=fixture_id,
                fixture_class=str(monolithic_row["instance_class"]),
                monolithic_row=dict(monolithic_row),
                boundary_rows=eta_rows,
                overwrite_rows=overwrite_rows,
            )
        )
    return tuple(links)


def run_joint_diagnostic() -> JsonDict:
    """Join p-bit boundary telemetry with overwrite-capable solver safety rows."""

    sources = load_joint_sources()
    links = build_shared_fixture_links(sources)
    joint_rows = _joint_rows(links)
    eta_summaries = _eta_summaries(links, joint_rows)
    eta_threshold = eta_threshold_from_summaries(eta_summaries)
    unsafe_false_accepts = sum(int(row["unsafe_false_accept"]) for row in joint_rows)
    post_projection_validity_rate = _rate(
        sum(row["projection_valid"] for row in joint_rows),
        len(joint_rows),
    )
    fallback_completeness_rate = _rate(
        sum(
            row["projection_valid"] and row["fallback_complete"] and row["final_matches_baseline"]
            for row in joint_rows
        ),
        len(joint_rows),
    )
    interpretable = bool(
        all(sources["source_readiness"].values())
        and len(links) == EXPECTED_SHARED_FIXTURE_COUNT
        and sorted(float(key) for key in eta_summaries) == list(ETA_VALUES)
        and eta_threshold is not None
        and post_projection_validity_rate == 1.0
        and fallback_completeness_rate == 1.0
        and unsafe_false_accepts == 0
        and joint_rows
    )
    return {
        "fixture_count": len(links),
        "shared_fixture_ids": list(sources["shared_fixture_ids"]),
        "eta_values": list(ETA_VALUES),
        "eta_threshold_estimate": eta_threshold,
        "solver_overwrite_enabled": True,
        "conflict_delta_vs_monolithic": max(
            summary["conflict_delta_vs_monolithic"] for summary in eta_summaries.values()
        ),
        "convergence_delta_vs_monolithic": max(
            summary["convergence_delta_vs_monolithic"] for summary in eta_summaries.values()
        ),
        "post_projection_validity_rate": post_projection_validity_rate,
        "fallback_completeness_rate": fallback_completeness_rate,
        "unsafe_false_accepts": unsafe_false_accepts,
        "pbit_boundary_overwrite_ready": interpretable,
        "comparison_variants": list(COMPARISON_VARIANTS),
        "source_readiness": dict(sources["source_readiness"]),
        "source_artifacts": list(sources["source_artifacts"]),
        "eta_summaries": eta_summaries,
        "variant_summaries": _variant_summaries(links, joint_rows, eta_summaries),
        "joint_results": joint_rows,
    }


def eta_threshold_from_summaries(
    eta_summaries: Mapping[str, Mapping[str, Any]],
) -> float | None:
    """Estimate the first tested eta that does not regress the monolithic run."""

    for eta in ETA_VALUES:
        summary = eta_summaries.get(str(eta))
        if (
            summary is not None
            and summary["conflict_delta_vs_monolithic"] >= 0
            and summary["convergence_delta_vs_monolithic"] >= 0
            and summary["unsafe_false_accepts"] == 0
        ):
            return eta
    return None


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build the terminal Exp5384 artifact from the CPU-only diagnostic."""

    diagnostic = run_joint_diagnostic()
    blockers = _readiness_blockers(diagnostic, tests_run)
    ready = bool(diagnostic["pbit_boundary_overwrite_ready"] and bool(tests_run) and not blockers)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "honest_blocked",
        "pbit_boundary_overwrite_ready": ready,
        "simulation_only": True,
        "hardware_speedup_claim": False,
        "fixture_count": diagnostic["fixture_count"],
        "eta_values": diagnostic["eta_values"],
        "eta_threshold_estimate": diagnostic["eta_threshold_estimate"],
        "solver_overwrite_enabled": diagnostic["solver_overwrite_enabled"],
        "conflict_delta_vs_monolithic": diagnostic["conflict_delta_vs_monolithic"],
        "convergence_delta_vs_monolithic": diagnostic["convergence_delta_vs_monolithic"],
        "post_projection_validity_rate": diagnostic["post_projection_validity_rate"],
        "fallback_completeness_rate": diagnostic["fallback_completeness_rate"],
        "unsafe_false_accepts": diagnostic["unsafe_false_accepts"],
        "honest_verdict": _honest_verdict(ready, diagnostic),
        "tests_run": [dict(row) for row in tests_run],
        "shared_fixture_ids": diagnostic["shared_fixture_ids"],
        "comparison_variants": diagnostic["comparison_variants"],
        "source_readiness": diagnostic["source_readiness"],
        "source_artifacts": diagnostic["source_artifacts"],
        "eta_summaries": diagnostic["eta_summaries"],
        "variant_summaries": diagnostic["variant_summaries"],
        "joint_results": diagnostic["joint_results"],
        "readiness_blockers": blockers,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "claim_limits": [
            "deterministic CPU-only joint diagnostic",
            "shared fixtures only; unrelated QSTR and repair rows are excluded",
            "p-bit boundary exchange explains sampler telemetry, not validity",
            "overwrite-capable solver guidance remains authoritative",
            "no hardware execution or hardware speedup claim",
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the validated Exp5384 artifact and return it."""

    artifact = build_artifact(tests_run=[] if tests_run is None else tests_run)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the joint diagnostic drifts from the safety contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    _require(artifact["field_principles"] == FIELD_PRINCIPLES, "field_principles")
    _require(artifact["status"] in {"complete", "honest_blocked"}, "status")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact["simulation_only"] is True, "simulation_only")
    _require(artifact["hardware_speedup_claim"] is False, "hardware_speedup_claim")
    _require(
        _is_bare_bool(artifact["pbit_boundary_overwrite_ready"]), "pbit_boundary_overwrite_ready"
    )
    _require(_is_bare_bool(artifact["solver_overwrite_enabled"]), "solver_overwrite_enabled")
    _require(_is_bare_int(artifact["fixture_count"]), "fixture_count")
    _require(_is_bare_int(artifact["unsafe_false_accepts"]), "unsafe_false_accepts")
    for field in (
        "conflict_delta_vs_monolithic",
        "convergence_delta_vs_monolithic",
        "post_projection_validity_rate",
        "fallback_completeness_rate",
    ):
        _require(_is_bare_numeric(artifact[field]), field)
    _require(list(artifact["eta_values"]) == list(ETA_VALUES), "eta_values")
    _require(
        artifact["eta_threshold_estimate"] is None
        or artifact["eta_threshold_estimate"] in artifact["eta_values"],
        "eta_threshold_estimate",
    )
    _require(isinstance(artifact["tests_run"], list), "tests_run")
    _require("REQ-VERIFY-5384" in artifact["spec_refs"], "spec_refs")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum")

    if artifact["pbit_boundary_overwrite_ready"]:
        _require(artifact["status"] == "complete", "status")
        _require(bool(artifact["tests_run"]), "tests_run")
        _require(artifact["fixture_count"] == EXPECTED_SHARED_FIXTURE_COUNT, "fixture_count")
        _require(artifact["solver_overwrite_enabled"] is True, "solver_overwrite_enabled")
        _require(artifact["post_projection_validity_rate"] == 1.0, "post_projection_validity_rate")
        _require(artifact["fallback_completeness_rate"] == 1.0, "fallback_completeness_rate")
        _require(artifact["unsafe_false_accepts"] == 0, "unsafe_false_accepts")
        _require(artifact["simulation_only"] is True, "simulation_only")
        _require(artifact["hardware_speedup_claim"] is False, "hardware_speedup_claim")
        _require(
            artifact["comparison_variants"] == list(COMPARISON_VARIANTS), "comparison_variants"
        )
        _validate_joint_rows(artifact["joint_results"])


def _joint_rows(links: Sequence[SharedFixtureLink]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for link in links:
        monolithic_conflicts = int(link.monolithic_row["cdcl_metrics"]["conflicts"])
        monolithic_steps = int(link.monolithic_row["sweeps_to_convergence"])
        for boundary_row in link.boundary_rows:
            boundary_delta = boundary_row["baseline_comparison"]
            for overwrite_row in link.overwrite_rows:
                rows.append(
                    {
                        "variant": "overwrite_guided",
                        "fixture_id": link.fixture_id,
                        "fixture_class": link.fixture_class,
                        "eta": float(boundary_row["eta"]),
                        "exchange_mode": boundary_row["exchange_mode"],
                        "hint_class": overwrite_row["hint_class"],
                        "solver_action": overwrite_row["solver_action"],
                        "monolithic_conflicts": monolithic_conflicts,
                        "boundary_conflicts": int(boundary_row["cdcl_metrics"]["conflicts"]),
                        "conflict_delta_vs_monolithic": int(boundary_delta["conflict_delta"]),
                        "overwrite_conflict_delta_vs_no_hint": int(
                            overwrite_row["conflict_delta_vs_no_hint"]
                        ),
                        "joint_conflict_delta_vs_monolithic": int(boundary_delta["conflict_delta"])
                        + int(overwrite_row["conflict_delta_vs_no_hint"]),
                        "monolithic_convergence_steps": monolithic_steps,
                        "boundary_convergence_steps": int(boundary_row["sweeps_to_convergence"]),
                        "convergence_delta_vs_monolithic": int(boundary_delta["convergence_delta"]),
                        "overwrite_convergence_delta_vs_no_hint": int(
                            overwrite_row["convergence_delta_vs_no_hint"]
                        ),
                        "joint_convergence_delta_vs_monolithic": int(
                            boundary_delta["convergence_delta"]
                        )
                        + int(overwrite_row["convergence_delta_vs_no_hint"]),
                        "projection_valid": bool(overwrite_row["projection_valid"]),
                        "fallback_complete": bool(overwrite_row["fallback_complete"]),
                        "final_matches_baseline": bool(overwrite_row["final_matches_baseline"]),
                        "unsafe_false_accept": bool(overwrite_row["unsafe_false_accept"]),
                        "solver_overwrite_enabled": True,
                        "simulation_only": True,
                        "hardware_speedup_claim": False,
                    }
                )
    return rows


def _eta_summaries(
    links: Sequence[SharedFixtureLink],
    joint_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    summaries: JsonDict = {}
    for eta in ETA_VALUES:
        boundary_rows = [row for link in links for row in link.boundary_rows if row["eta"] == eta]
        eta_joint_rows = [row for row in joint_rows if row["eta"] == eta]
        projection_validity_rate = _rate(
            sum(row["projection_valid"] for row in eta_joint_rows),
            len(eta_joint_rows),
        )
        fallback_completeness_rate = _rate(
            sum(
                row["projection_valid"]
                and row["fallback_complete"]
                and row["final_matches_baseline"]
                for row in eta_joint_rows
            ),
            len(eta_joint_rows),
        )
        summaries[str(eta)] = {
            "row_count": len(eta_joint_rows),
            "shared_fixture_count": len(boundary_rows),
            "conflict_delta_vs_monolithic": sum(
                int(row["baseline_comparison"]["conflict_delta"]) for row in boundary_rows
            ),
            "convergence_delta_vs_monolithic": sum(
                int(row["baseline_comparison"]["convergence_delta"]) for row in boundary_rows
            ),
            "overwrite_conflict_delta_vs_no_hint": sum(
                int(row["overwrite_conflict_delta_vs_no_hint"]) for row in eta_joint_rows
            ),
            "overwrite_convergence_delta_vs_no_hint": sum(
                int(row["overwrite_convergence_delta_vs_no_hint"]) for row in eta_joint_rows
            ),
            "joint_conflict_delta_vs_monolithic": sum(
                int(row["joint_conflict_delta_vs_monolithic"]) for row in eta_joint_rows
            ),
            "joint_convergence_delta_vs_monolithic": sum(
                int(row["joint_convergence_delta_vs_monolithic"]) for row in eta_joint_rows
            ),
            "post_projection_validity_rate": projection_validity_rate,
            "fallback_completeness_rate": fallback_completeness_rate,
            "unsafe_false_accepts": sum(int(row["unsafe_false_accept"]) for row in eta_joint_rows),
        }
    return summaries


def _variant_summaries(
    links: Sequence[SharedFixtureLink],
    joint_rows: Sequence[Mapping[str, Any]],
    eta_summaries: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    monolithic_conflicts = sum(
        int(link.monolithic_row["cdcl_metrics"]["conflicts"]) for link in links
    )
    monolithic_steps = sum(int(link.monolithic_row["sweeps_to_convergence"]) for link in links)
    return {
        "monolithic_pbit": {
            "row_count": len(links),
            "conflicts": monolithic_conflicts,
            "convergence_steps": monolithic_steps,
        },
        "boundary_exchange": {
            "row_count": len(links) * len(ETA_VALUES),
            "eta_values": list(ETA_VALUES),
            "eta_threshold_estimate": eta_threshold_from_summaries(eta_summaries),
        },
        "overwrite_guided": {
            "row_count": len(joint_rows),
            "post_projection_validity_rate": _rate(
                sum(row["projection_valid"] for row in joint_rows),
                len(joint_rows),
            ),
            "fallback_completeness_rate": _rate(
                sum(
                    row["projection_valid"]
                    and row["fallback_complete"]
                    and row["final_matches_baseline"]
                    for row in joint_rows
                ),
                len(joint_rows),
            ),
            "unsafe_false_accepts": sum(int(row["unsafe_false_accept"]) for row in joint_rows),
        },
    }


def _readiness_blockers(
    diagnostic: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[str]:
    checks = (
        (
            not all(diagnostic["source_readiness"].values()),
            "source_artifact_not_ready",
        ),
        (
            diagnostic["fixture_count"] != EXPECTED_SHARED_FIXTURE_COUNT,
            "shared_fixture_count_mismatch",
        ),
        (
            list(diagnostic["eta_values"]) != list(ETA_VALUES),
            "eta_values_mismatch",
        ),
        (
            diagnostic["eta_threshold_estimate"] is None,
            "eta_threshold_missing",
        ),
        (
            diagnostic["solver_overwrite_enabled"] is not True,
            "solver_overwrite_disabled",
        ),
        (
            diagnostic["post_projection_validity_rate"] != 1.0,
            "post_projection_validity_incomplete",
        ),
        (
            diagnostic["fallback_completeness_rate"] != 1.0,
            "fallback_completeness_incomplete",
        ),
        (diagnostic["unsafe_false_accepts"] != 0, "unsafe_false_accepts"),
        (
            not diagnostic["pbit_boundary_overwrite_ready"],
            "joint_results_not_interpretable",
        ),
        (not tests_run, "tests_not_recorded"),
    )
    return [name for failed, name in checks if failed]


def _honest_verdict(ready: bool, diagnostic: Mapping[str, Any]) -> str:
    if not ready:
        return "blocked_pbit_boundary_overwrite_joint_not_ready"
    return (
        "complete: CPU joint diagnostic found boundary exchange explains "
        f"conflict/convergence deltas at eta >= {diagnostic['eta_threshold_estimate']} "
        "while overwrite-capable solver guidance preserved validity and fallback "
        "completeness with no hardware speedup claim"
    )


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": artifact["experiment_id"],
        "spec_refs": artifact["spec_refs"],
        "tests_run": artifact["tests_run"],
        "metrics": {
            field: artifact[field]
            for field in REQUIRED_ARTIFACT_FIELDS
            if field != "honest_verdict"
        },
        "shared_fixture_ids": artifact["shared_fixture_ids"],
        "eta_summaries": artifact["eta_summaries"],
        "variant_summaries": artifact["variant_summaries"],
        "joint_results": artifact["joint_results"],
        "source_artifacts": artifact["source_artifacts"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_joint_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    _require(
        len(rows) == EXPECTED_SHARED_FIXTURE_COUNT * len(ETA_VALUES) * len(HINT_CLASS_NAMES),
        "joint_results",
    )
    for row in rows:
        _require(row["variant"] == "overwrite_guided", "row variant")
        _require(row["solver_overwrite_enabled"] is True, "row solver_overwrite_enabled")
        _require(row["projection_valid"] is True, "row projection_valid")
        _require(row["fallback_complete"] is True, "row fallback_complete")
        _require(row["final_matches_baseline"] is True, "row final_matches_baseline")
        _require(row["unsafe_false_accept"] is False, "row unsafe_false_accept")
        _require(row["simulation_only"] is True, "row simulation_only")
        _require(row["hardware_speedup_claim"] is False, "row hardware_speedup_claim")


def _load_json(relative_path: Path) -> JsonDict:
    return json.loads((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _rate(numerator: int | float, denominator: int) -> float:
    return 1.0 if denominator == 0 else float(numerator) / denominator


def _is_bare_bool(value: Any) -> bool:
    return type(value) is bool


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def _is_bare_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
