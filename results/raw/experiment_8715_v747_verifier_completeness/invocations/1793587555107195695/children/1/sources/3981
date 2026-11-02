"""Exp 3696 re-ships the second-pair detector with a code operating point.

The runner is intentionally small: Exp 3695 already established that a
code-native AST/runtime signal beats chance.  This module checks that gate, runs
the shipped detector surface with that code path wired, and records whether the
math operating point stayed intact while code now carries signal.

Spec: REQ-SPOE-3696, SCENARIO-SPOE-3696.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from carnot.pipeline import second_pair_detector as spd


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3696_reship_detector_math_plus_code.json")
EXP3695_REL_PATH = Path("results/experiment_3695_code_native_verifier.json")
BASELINE_SHIP_REL_PATH = Path("results/experiment_3671_ship_second_pair_of_eyes_detector.json")
RANDOM_SEED = 3696
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached corpora; no LLM load; no compute-bound marker)."
)

VERDICT_RESHIPPED = "complete: detector_reshipped_math_plus_code_operating_point_e2e_green"
VERDICT_BLOCKED = "complete: blocked_code_signal_not_recovered_or_module_unavailable"
TERMINAL_VERDICTS = (VERDICT_RESHIPPED, VERDICT_BLOCKED)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "module_code_path_updated",
    "math_operating_point_unchanged",
    "code_operating_point_auroc",
    "code_operating_point_calibration",
    "e2e_test_passed",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "module_code_path_updated": (
        "True iff the shipped module now wires the code-native signal into a "
        "calibrated code operating point."
    ),
    "math_operating_point_unchanged": (
        "True iff the strong math operating point (AUROC 0.98, ECE 0.009) is "
        "preserved -- the code add must not regress math."
    ),
    "code_operating_point_auroc": (
        "The code AUROC the shipped detector now achieves -- the deployable code number."
    ),
    "code_operating_point_calibration": (
        "Brier/ECE of the shipped code operating point -- a deployable point must be calibrated."
    ),
    "e2e_test_passed": (
        "True iff the shipped-surface E2E test passes after the change."
    ),
    "adversarial_verify_clean": "True iff no critical flag.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3696 artifact from cached corpora and prior gates."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    preconditions = check_preconditions(root_path)
    if not all(bool(item["available"]) for item in preconditions):
        return build_artifact_from_measurements(
            blocked=True,
            ship_artifact={},
            baseline_ship_artifact={},
            module_code_path_updated=False,
            e2e_test_passed=False,
            adversarial_verify_clean=False,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            preconditions_checked=preconditions,
        )

    ship_artifact = spd.build_ship_artifact(
        root_path,
        output_path=spd.SHIP_OUTPUT_REL_PATH,
        tests_run=tests_run,
    )
    baseline_ship_artifact = _read_json_object(root_path / BASELINE_SHIP_REL_PATH)
    return build_artifact_from_measurements(
        blocked=False,
        ship_artifact=ship_artifact,
        baseline_ship_artifact=baseline_ship_artifact,
        module_code_path_updated=module_code_path_updated(),
        e2e_test_passed=bool(ship_artifact.get("e2e_test_passed")),
        adversarial_verify_clean=True,
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
        preconditions_checked=preconditions,
        extra={
            "ship_artifact_summary": _ship_summary(ship_artifact),
            "exp3695_code_operating_point": dict(
                getattr(spd, "CODE_NATIVE_OPERATING_POINT", {})
            ),
        },
    )


def build_artifact_from_measurements(
    *,
    blocked: bool,
    ship_artifact: Mapping[str, Any],
    baseline_ship_artifact: Mapping[str, Any],
    module_code_path_updated: bool,
    e2e_test_passed: bool,
    adversarial_verify_clean: bool,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    preconditions_checked: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the terminal artifact from measured shipped-surface metrics."""

    start = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    code_auroc = None if blocked else _domain_value(
        ship_artifact,
        "fused_detector_auroc_per_domain",
        "code",
    )
    code_calibration = (
        {}
        if blocked
        else _domain_mapping(ship_artifact, "calibration_brier_ece_per_domain", "code")
    )
    math_unchanged = False if blocked else math_operating_point_unchanged(
        ship_artifact,
        baseline_ship_artifact,
    )
    acceptance_passed = bool(
        module_code_path_updated and math_unchanged and e2e_test_passed
    )
    code_has_signal = code_auroc is not None and float(code_auroc) > 0.5
    verdict = (
        VERDICT_RESHIPPED
        if not blocked and acceptance_passed and code_has_signal and adversarial_verify_clean
        else VERDICT_BLOCKED
    )
    artifact: JsonDict = {
        "artifact": "experiment_3696_reship_detector_math_plus_code",
        "schema": "carnot.reship_detector_math_plus_code_3696.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "module_code_path_updated": bool(module_code_path_updated),
        "math_operating_point_unchanged": bool(math_unchanged),
        "code_operating_point_auroc": None if code_auroc is None else _round(float(code_auroc)),
        "code_operating_point_calibration": dict(code_calibration),
        "e2e_test_passed": bool(e2e_test_passed),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _round(max(0.0, finished - start)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "module_code_path_updated == true AND "
                "math_operating_point_unchanged == true AND "
                "e2e_test_passed == true"
            ),
            "passed": acceptance_passed,
            "principle": (
                "Re-shipping the detector with a code operating point is correct "
                "only if the code path is wired, the math point is preserved, and "
                "the E2E passes -- otherwise the change is not deployable."
            ),
        },
        "preconditions_checked": [dict(item) for item in preconditions_checked or []],
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "math_operating_point": _math_operating_point(ship_artifact, baseline_ship_artifact),
        "code_operating_point": _code_operating_point(ship_artifact),
    }
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def module_code_path_updated() -> bool:
    """Return true when the shipped module exposes the Exp 3695 code path."""

    point = getattr(spd, "CODE_NATIVE_OPERATING_POINT", {})
    return bool(
        getattr(spd, "CODE_NATIVE_CODE_PATH_ENABLED", False) is True
        and isinstance(point, Mapping)
        and point.get("source") == str(EXP3695_REL_PATH)
        and callable(getattr(spd, "_score_code_native_rows", None))
    )


def math_operating_point_unchanged(
    ship_artifact: Mapping[str, Any],
    baseline_ship_artifact: Mapping[str, Any],
) -> bool:
    """Return true when math AUROC remains at 0.98 and calibration does not regress."""

    updated_auroc = _domain_value(ship_artifact, "fused_detector_auroc_per_domain", "math")
    updated_calibration = _domain_mapping(
        ship_artifact,
        "calibration_brier_ece_per_domain",
        "math",
    )
    if updated_auroc is None or not updated_calibration:
        return False
    baseline_auroc = _domain_value(
        baseline_ship_artifact,
        "fused_detector_auroc_per_domain",
        "math",
    )
    baseline_calibration = _domain_mapping(
        baseline_ship_artifact,
        "calibration_brier_ece_per_domain",
        "math",
    )
    auroc_preserved = round(float(updated_auroc), 2) == 0.98
    if baseline_auroc is not None:
        auroc_preserved = auroc_preserved and abs(
            float(updated_auroc) - float(baseline_auroc)
        ) <= 1e-6
    ece_ok = float(updated_calibration.get("ece", math.inf)) <= 0.009
    brier_ok = True
    if baseline_calibration:
        ece_ok = ece_ok and float(updated_calibration.get("ece", math.inf)) <= (
            float(baseline_calibration.get("ece", math.inf)) + 1e-6
        )
        brier_ok = float(updated_calibration.get("brier", math.inf)) <= (
            float(baseline_calibration.get("brier", math.inf)) + 1e-5
        )
    return bool(auroc_preserved and ece_ok and brier_ok)


def check_preconditions(root: Path) -> list[JsonDict]:
    """Check the Exp 3695 gate and shipped detector module importability."""

    exp3695 = _read_json_object(root / EXP3695_REL_PATH)
    checks = [
        {
            "resource": "exp3695_code_signal_recovered",
            "available": exp3695.get("code_signal_recovered") is True,
            "detail": str(root / EXP3695_REL_PATH),
        },
        {
            "resource": "second_pair_detector_module",
            "available": bool(
                callable(getattr(spd, "score_candidates", None))
                and callable(getattr(spd, "load_cached_labeled_examples", None))
            ),
            "detail": spd.DETECTOR_MODULE_PATH,
        },
    ]
    return checks


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3696 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError("honest_verdict is not an accepted Exp 3696 terminal verdict")
    for field in (
        "module_code_path_updated",
        "math_operating_point_unchanged",
        "e2e_test_passed",
        "adversarial_verify_clean",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    code_auroc = artifact.get("code_operating_point_auroc")
    if code_auroc is not None and not isinstance(code_auroc, (int, float)):
        raise ValueError("code_operating_point_auroc must be a number or null")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3696 artifact fields."""

    payload = {
        "module_code_path_updated": artifact.get("module_code_path_updated"),
        "math_operating_point_unchanged": artifact.get("math_operating_point_unchanged"),
        "code_operating_point_auroc": artifact.get("code_operating_point_auroc"),
        "code_operating_point_calibration": artifact.get(
            "code_operating_point_calibration"
        ),
        "e2e_test_passed": artifact.get("e2e_test_passed"),
        "adversarial_verify_clean": artifact.get("adversarial_verify_clean"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, adversarial-check, validate, and persist the Exp 3696 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = run_adversarial_verify_report(output)
    artifact["adversarial_verify_report"] = compact_adversarial_report(report)
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_artifact_from_measurements(
    root: Path | str,
    *,
    output_path: Path | str,
    **kwargs: Any,
) -> Path:
    """Persist a synthetic or pre-measured Exp 3696 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact_from_measurements(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_adversarial_verify_report(path: Path) -> JsonDict:
    """Run scripts/adversarial_verify.py against an artifact path."""

    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3696", verifier_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(path))


def compact_adversarial_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep the adversarial report small and deterministic in the artifact."""

    flags = [dict(flag) for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    return {"flag_count": len(flags), "flags": flags}


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true when no adversarial flag is critical."""

    flags = report.get("flags", [])
    if not isinstance(flags, Sequence):
        return False
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def _ship_summary(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "fused_detector_auroc_per_domain": artifact.get("fused_detector_auroc_per_domain"),
        "calibration_brier_ece_per_domain": artifact.get(
            "calibration_brier_ece_per_domain"
        ),
        "operating_points": artifact.get("operating_points"),
        "n_examples_per_domain": artifact.get("n_examples_per_domain"),
    }


def _math_operating_point(
    ship_artifact: Mapping[str, Any],
    baseline_ship_artifact: Mapping[str, Any],
) -> JsonDict:
    return {
        "updated": {
            "auroc": _domain_value(ship_artifact, "fused_detector_auroc_per_domain", "math"),
            "calibration": _domain_mapping(
                ship_artifact,
                "calibration_brier_ece_per_domain",
                "math",
            ),
            "operating_point": _domain_mapping(ship_artifact, "operating_points", "math"),
        },
        "baseline_exp3671": {
            "auroc": _domain_value(
                baseline_ship_artifact,
                "fused_detector_auroc_per_domain",
                "math",
            ),
            "calibration": _domain_mapping(
                baseline_ship_artifact,
                "calibration_brier_ece_per_domain",
                "math",
            ),
            "operating_point": _domain_mapping(
                baseline_ship_artifact,
                "operating_points",
                "math",
            ),
        },
    }


def _code_operating_point(ship_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "auroc": _domain_value(ship_artifact, "fused_detector_auroc_per_domain", "code"),
        "calibration": _domain_mapping(
            ship_artifact,
            "calibration_brier_ece_per_domain",
            "code",
        ),
        "operating_point": _domain_mapping(ship_artifact, "operating_points", "code"),
        "recall_at_fixed_fpr": _domain_mapping(
            ship_artifact,
            "recall_at_fixed_fpr_table",
            "code",
        ),
    }


def _domain_value(
    artifact: Mapping[str, Any],
    field: str,
    domain: str,
) -> float | None:
    rows = artifact.get(field)
    if not isinstance(rows, Mapping):
        return None
    value = rows.get(domain)
    return None if value is None else float(value)


def _domain_mapping(
    artifact: Mapping[str, Any],
    field: str,
    domain: str,
) -> JsonDict:
    rows = artifact.get(field)
    if not isinstance(rows, Mapping):
        return {}
    value = rows.get(domain)
    return dict(value) if isinstance(value, Mapping) else {}


def _read_json_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


__all__ = [
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "build_artifact_from_measurements",
    "check_preconditions",
    "math_operating_point_unchanged",
    "module_code_path_updated",
    "validate_artifact",
    "write_artifact",
    "write_artifact_from_measurements",
]
