#!/usr/bin/env python3
"""Exp 3717: full G4 headline provenance audit.

Spec: REQ-PUBLISH-3717, SCENARIO-PUBLISH-3717.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = _SCRIPT_PATH.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.adversarial_verify import verify_artifact


NORTH_STAR_REL_PATH = Path("ops/north-star.md")
OUTPUT_REL_PATH = Path("results/experiment_3717_g4_full_provenance_audit.json")

G1_FOVER_REL_PATH = Path("results/experiment_2850_fover_dual_condition_integrity_v4.json")
FOVER_MEMORY_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
DETECTOR_REL_PATH = Path("results/experiment_3706_reconcile_shipped_detector_heldout.json")

DEFAULT_RANDOM_SEED = 3717
SUCCESS_VERDICT = "complete: g4_fully_traced_every_headline_number_to_clean_primary_artifact"
GAP_VERDICT = "complete: g4_provenance_gap_found_operator_action_items_recorded"
BLOCKED_VERDICT = "complete: blocked_primary_artifact_unavailable"
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: reads north-star + primary "
    "artifacts; no live inference; no compute-bound marker)."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "provenance_rows",
    "n_numbers_audited",
    "all_numbers_trace_to_clean_artifacts",
    "any_cited_source_flagged",
    "g4_status",
    "operator_action_items",
    "north_star_unmodified_assert",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "provenance_rows": (
        "Per-number {headline_number, value, primary_artifact, has_seed, "
        "has_checksum, adversarial_clean} -- the end-to-end G4 trace."
    ),
    "n_numbers_audited": (
        "How many headline + north-star numbers were traced (sample-size of the audit)."
    ),
    "all_numbers_trace_to_clean_artifacts": (
        "BARE bool. True iff every audited number traces to a primary artifact "
        "with seed + checksum AND a clean adversarial re-check. STORE AS BARE true/false."
    ),
    "any_cited_source_flagged": (
        "BARE bool. True iff any cited primary artifact carries flagged_adversarial "
        "(the exp3704 lesson) -> an operator corrigendum item. STORE AS BARE true/false."
    ),
    "g4_status": "fully_traced / gap_found -- the hardened G4 verdict.",
    "operator_action_items": (
        "Any flagged-source corrigenda or missing-seed/checksum gaps for OPERATOR "
        "action (read-only audit; no auto-fix)."
    ),
    "north_star_unmodified_assert": (
        "Asserts ops/north-star.md was NOT edited (operator-curated)."
    ),
    "adversarial_verify_clean": (
        "True iff THIS audit artifact passes adversarial_verify with no critical flag."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}

NUMBER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "headline_number": "FoVer AUROC",
        "value": 0.9131,
        "primary_artifact": G1_FOVER_REL_PATH,
        "field_path": ("condition_a_production_auroc_mean",),
        "tolerance": 0.00005,
    },
    {
        "headline_number": "FoVer CI95 low",
        "value": 0.9027,
        "primary_artifact": FOVER_MEMORY_REL_PATH,
        "field_path": ("condition_a_production_auroc_ci95", "low"),
        "tolerance": 0.00005,
    },
    {
        "headline_number": "FoVer CI95 high",
        "value": 0.9235,
        "primary_artifact": FOVER_MEMORY_REL_PATH,
        "field_path": ("condition_a_production_auroc_ci95", "high"),
        "tolerance": 0.00005,
    },
    {
        "headline_number": "FR-11 AUROC contribution",
        "value": 0.0185,
        "primary_artifact": FOVER_MEMORY_REL_PATH,
        "field_path": ("learning_contribution_ci95", "mean"),
        "tolerance": 0.00005,
    },
    {
        "headline_number": "FR-11 contribution CI95 low",
        "value": 0.0125,
        "primary_artifact": FOVER_MEMORY_REL_PATH,
        "field_path": ("learning_contribution_ci95", "low"),
        "tolerance": 0.00005,
    },
    {
        "headline_number": "FR-11 contribution CI95 high",
        "value": 0.0245,
        "primary_artifact": FOVER_MEMORY_REL_PATH,
        "field_path": ("learning_contribution_ci95", "high"),
        "tolerance": 0.00005,
    },
    {
        "headline_number": "shipped detector math operating point AUROC and ECE",
        "value": {"auroc": 0.98, "ece": 0.009},
        "primary_artifact": DETECTOR_REL_PATH,
        "field_path": (
            ("math_operating_point", "auroc"),
            ("math_operating_point", "calibration", "ece"),
        ),
        "tolerance": {"auroc": 0.005, "ece": 0.001},
    },
)


def audit_g4(
    *,
    repo_root: Path = REPO_ROOT,
    verifier: Callable[[Path], Mapping[str, Any]] = verify_artifact,
    started_s: float | None = None,
    now_s: float | None = None,
    self_adversarial_verify_clean: bool = True,
) -> dict[str, Any]:
    """Build the Exp 3717 G4 audit artifact without editing source documents."""

    start = time.monotonic() if started_s is None else float(started_s)
    north_star_path = repo_root / NORTH_STAR_REL_PATH
    north_star_before = _sha256_file(north_star_path)
    if not (repo_root / G1_FOVER_REL_PATH).exists():
        artifact = _base_artifact(
            honest_verdict=BLOCKED_VERDICT,
            g4_status="blocked",
            rows=[],
            actions=[f"primary FoVer artifact unavailable: {G1_FOVER_REL_PATH}"],
            any_flagged=False,
            north_star_unmodified=north_star_before == _sha256_file(north_star_path),
            self_clean=False,
            started_s=start,
            now_s=now_s,
        )
        validate_artifact(artifact)
        return artifact

    payloads = _load_payloads(repo_root)
    reports = {
        rel_path: verifier(repo_root / rel_path)
        for rel_path, payload in payloads.items()
        if payload is not None
    }
    rows = [_provenance_row(spec, payloads, reports) for spec in NUMBER_SPECS]
    actions = _operator_action_items(rows)
    any_flagged = any(
        _payload_flagged(payload) for payload in payloads.values() if payload is not None
    ) or any(_report_has_flagged_adversarial(report) for report in reports.values())
    all_clean = bool(rows) and all(
        row["has_seed"]
        and row["has_checksum"]
        and row["adversarial_clean"]
        and row["value_matches_artifact"]
        for row in rows
    )
    verdict = SUCCESS_VERDICT if all_clean else GAP_VERDICT
    status = "fully_traced" if all_clean else "gap_found"
    artifact = _base_artifact(
        honest_verdict=verdict,
        g4_status=status,
        rows=rows,
        actions=actions,
        any_flagged=any_flagged,
        north_star_unmodified=north_star_before == _sha256_file(north_star_path),
        self_clean=self_adversarial_verify_clean,
        started_s=start,
        now_s=now_s,
    )
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3717 audit schema."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if not str(artifact["inference_substrate"]).startswith("aggregation_from_upstream_artifacts"):
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if any(marker in str(artifact["inference_substrate"]) for marker in ("GGUF", "CUDA")):
        raise ValueError("inference_substrate must not contain GGUF/CUDA markers")
    if artifact["honest_verdict"] not in {SUCCESS_VERDICT, GAP_VERDICT, BLOCKED_VERDICT}:
        raise ValueError("unsupported honest_verdict")
    if artifact["g4_status"] not in {"fully_traced", "gap_found", "blocked"}:
        raise ValueError("g4_status must be fully_traced, gap_found, or blocked")
    for field in (
        "all_numbers_trace_to_clean_artifacts",
        "any_cited_source_flagged",
        "north_star_unmodified_assert",
        "adversarial_verify_clean",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare boolean")
    rows = artifact["provenance_rows"]
    if not isinstance(rows, list):
        raise ValueError("provenance_rows must be a list")
    if artifact["n_numbers_audited"] != len(rows):
        raise ValueError("n_numbers_audited must equal provenance_rows length")
    for row in rows:
        _validate_row(row)
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    gate = artifact.get("acceptance_gate")
    if not isinstance(gate, Mapping) or not {"condition", "principle", "passed"} <= set(gate):
        raise ValueError("acceptance_gate must include condition, principle, and passed")


def write_artifact_with_self_check(
    *,
    repo_root: Path = REPO_ROOT,
    verifier: Callable[[Path], Mapping[str, Any]] = verify_artifact,
) -> dict[str, Any]:
    """Write the audit artifact and stamp the live self-verifier result."""

    artifact = audit_g4(repo_root=repo_root, verifier=verifier)
    out_path = repo_root / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_path, artifact)
    self_report = verifier(out_path)
    artifact["adversarial_verify_clean"] = _report_clean(self_report)
    artifact["self_adversarial_verify_report"] = dict(self_report)
    artifact["acceptance_gate"]["passed"] = _acceptance_passed(artifact)
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    _write_json(out_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. No arguments are currently needed."""

    if argv not in (None, []):
        raise SystemExit("experiment_3717 accepts no arguments")
    artifact = write_artifact_with_self_check(repo_root=REPO_ROOT, verifier=verify_artifact)
    print(artifact["honest_verdict"])
    return 0


def _base_artifact(
    *,
    honest_verdict: str,
    g4_status: str,
    rows: list[dict[str, Any]],
    actions: list[str],
    any_flagged: bool,
    north_star_unmodified: bool,
    self_clean: bool,
    started_s: float,
    now_s: float | None,
) -> dict[str, Any]:
    all_traced = bool(rows) and all(
        row["has_seed"]
        and row["has_checksum"]
        and row["adversarial_clean"]
        and row["value_matches_artifact"]
        for row in rows
    )
    artifact: dict[str, Any] = {
        "artifact": "experiment_3717_g4_full_provenance_audit",
        "schema": "carnot.g4_full_provenance_audit.v1",
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "provenance_rows": rows,
        "n_numbers_audited": len(rows),
        "all_numbers_trace_to_clean_artifacts": bool(all_traced),
        "any_cited_source_flagged": bool(any_flagged),
        "g4_status": g4_status,
        "operator_action_items": actions,
        "north_star_unmodified_assert": bool(north_star_unmodified),
        "adversarial_verify_clean": bool(self_clean),
        "random_seed": DEFAULT_RANDOM_SEED,
        "duration_s": _duration(started_s, now_s),
        "acceptance_gate": {
            "condition": (
                "provenance_rows present AND n_numbers_audited present AND "
                "all_numbers_trace_to_clean_artifacts present AND "
                "adversarial_verify_clean == true AND "
                "north_star_unmodified_assert == true"
            ),
            "principle": (
                "A trustworthy G4 audit traces every number to a clean primary "
                "artifact, is itself adversarial-clean, and leaves the "
                "operator-curated north-star untouched."
            ),
            "passed": False,
        },
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["acceptance_gate"]["passed"] = _acceptance_passed(artifact)
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _load_payloads(repo_root: Path) -> dict[Path, dict[str, Any] | None]:
    paths = {spec["primary_artifact"] for spec in NUMBER_SPECS}
    return {
        rel_path: _load_json(repo_root / rel_path) if (repo_root / rel_path).exists() else None
        for rel_path in paths
    }


def _provenance_row(
    spec: Mapping[str, Any],
    payloads: Mapping[Path, dict[str, Any] | None],
    reports: Mapping[Path, Mapping[str, Any]],
) -> dict[str, Any]:
    rel_path = spec["primary_artifact"]
    payload = payloads.get(rel_path)
    actual = _actual_value(payload, spec["field_path"]) if payload is not None else None
    report = reports.get(rel_path, {"flags": [], "loaded": False})
    stamped_flagged = _payload_flagged(payload) if payload is not None else False
    return {
        "headline_number": spec["headline_number"],
        "value": spec["value"],
        "artifact_value": actual,
        "primary_artifact": str(rel_path),
        "artifact_field_path": _field_path_label(spec["field_path"]),
        "has_seed": _has_seed(payload),
        "has_checksum": _has_checksum(payload),
        "adversarial_clean": bool(_report_clean(report) and not stamped_flagged),
        "flagged_adversarial_stamped": bool(stamped_flagged),
        "value_matches_artifact": _value_matches(spec["value"], actual, spec["tolerance"]),
        "live_adversarial_verify_report": dict(report),
    }


def _operator_action_items(rows: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for row in rows:
        label = f"{row['headline_number']} via {row['primary_artifact']}"
        if not row["has_seed"]:
            actions.append(f"missing random_seed/random_seeds_used: {label}")
        if not row["has_checksum"]:
            actions.append(f"missing reproducibility_checksum: {label}")
        if row["flagged_adversarial_stamped"]:
            actions.append(f"flagged_adversarial cited source requires corrigendum: {label}")
        if not row["adversarial_clean"]:
            actions.append(f"adversarial_verify not clean: {label}")
        if not row["value_matches_artifact"]:
            actions.append(f"value mismatch: {label}")
    return actions


def _validate_row(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("row must be a mapping")
    required = {
        "headline_number",
        "value",
        "primary_artifact",
        "has_seed",
        "has_checksum",
        "adversarial_clean",
    }
    if not required <= set(row):
        raise ValueError("row missing required provenance fields")
    for field in ("has_seed", "has_checksum", "adversarial_clean"):
        if type(row[field]) is not bool:
            raise ValueError(f"row {field} must be a bare boolean")


def _actual_value(payload: Mapping[str, Any], field_path: Any) -> Any:
    if isinstance(field_path[0], tuple):
        return {"auroc": _deep_get(payload, field_path[0]), "ece": _deep_get(payload, field_path[1])}
    return _deep_get(payload, field_path)


def _deep_get(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        current = current[key] if isinstance(current, Mapping) and key in current else None
    return current


def _value_matches(expected: Any, actual: Any, tolerance: Any) -> bool:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return all(
            _number_close(expected[key], actual.get(key), tolerance[key])
            for key in expected
        )
    return _number_close(expected, actual, tolerance)


def _number_close(expected: Any, actual: Any, tolerance: float) -> bool:
    return (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and abs(float(actual) - float(expected)) <= float(tolerance)
    )


def _field_path_label(field_path: Any) -> Any:
    if isinstance(field_path[0], tuple):
        return [".".join(path) for path in field_path]
    return ".".join(field_path)


def _has_seed(payload: Mapping[str, Any] | None) -> bool:
    return bool(
        payload is not None
        and (
            "random_seed" in payload
            or "random_seeds_used" in payload
            or "n_seeds" in payload
        )
    )


def _has_checksum(payload: Mapping[str, Any] | None) -> bool:
    return bool(payload is not None and "reproducibility_checksum" in payload)


def _payload_flagged(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("flagged_adversarial") is True)


def _report_clean(report: Mapping[str, Any]) -> bool:
    return not any(
        str(flag.get("severity", "")).lower() == "critical"
        for flag in report.get("flags", [])
        if isinstance(flag, Mapping)
    )


def _report_has_flagged_adversarial(report: Mapping[str, Any]) -> bool:
    return any(
        str(flag.get("kind", "")).lower() == "flagged_adversarial"
        for flag in report.get("flags", [])
        if isinstance(flag, Mapping)
    )


def _acceptance_passed(artifact: Mapping[str, Any]) -> bool:
    return bool(
        artifact.get("provenance_rows")
        and artifact.get("n_numbers_audited")
        and type(artifact.get("all_numbers_trace_to_clean_artifacts")) is bool
        and artifact.get("all_numbers_trace_to_clean_artifacts") is True
        and artifact.get("adversarial_verify_clean") is True
        and artifact.get("north_star_unmodified_assert") is True
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.monotonic() if now_s is None else float(now_s)
    return round(max(0.0001, end - started_s), 6)


def _checksum(artifact: Mapping[str, Any]) -> str:
    stable = {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    blob = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
