"""Exp 3706 reconciles the shipped detector against the held-out code audit.

Exp 3696 shipped a code operating point derived from Exp 3695's in-corpus
AUROC=1.0.  Exp 3705 then leak-audited that number.  This module updates the
product-facing claim either by recalibrating to the held-out number or by
narrowing the shipped detector back to math-only with an explicit no-code
verdict for code candidates.

Spec: REQ-SPOE-3706, SCENARIO-SPOE-3706.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from carnot.pipeline import second_pair_detector as spd


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3706_reconcile_shipped_detector_heldout.json")
EXP3705_REL_PATH = Path("results/experiment_3705_code_native_leak_audit_heldout.json")
RANDOM_SEED = 3706
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: scores cached corpora; no LLM load; no compute-bound marker)."
)

ACTION_RECALIBRATED = "recalibrated_to_heldout"
ACTION_NARROWED = "narrowed_to_math_only_abstain"
ACTION_BLOCKED = "blocked"

VERDICT_RECALIBRATED = "complete: shipped_detector_code_recalibrated_to_heldout_e2e_green"
VERDICT_NARROWED = (
    "complete: shipped_detector_narrowed_to_math_only_abstain_on_code_e2e_green"
)
VERDICT_BLOCKED = "complete: blocked_heldout_audit_unavailable"
TERMINAL_VERDICTS = (VERDICT_RECALIBRATED, VERDICT_NARROWED, VERDICT_BLOCKED)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "reconciliation_action",
    "shipped_code_operating_point_auroc",
    "math_operating_point_unchanged",
    "overclaim_removed",
    "e2e_test_passed",
    "operating_envelope_docstring_updated",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": INFERENCE_SUBSTRATE,
    "reconciliation_action": (
        "recalibrated_to_heldout / narrowed_to_math_only_abstain -- which "
        "branch the shipped surface took, traceable to Exp 3705's verdict."
    ),
    "shipped_code_operating_point_auroc": (
        "The code AUROC the shipped detector now claims -- the HELD-OUT number "
        "if survived, else null (abstain)."
    ),
    "math_operating_point_unchanged": (
        "True iff the strong math operating point (AUROC ~0.98, ECE ~0.009) "
        "is preserved -- the reconciliation must not regress math."
    ),
    "overclaim_removed": (
        "BARE bool. True iff the shipped detector no longer claims the inflated "
        "in-corpus 1.0 -- the product-integrity fix."
    ),
    "e2e_test_passed": "True iff the shipped-surface E2E test passes after the change.",
    "operating_envelope_docstring_updated": (
        "True iff the module docstring states the honest math (+code-or-abstain) "
        "operating points."
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
    """Build the Exp 3706 artifact from Exp 3705 and the shipped surface."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    audit_path = root_path / EXP3705_REL_PATH
    audit_artifact = _read_json_object(audit_path)
    module_available = detector_module_available()
    preconditions = [
        {
            "resource": "exp3705_heldout_audit",
            "available": bool(audit_artifact),
            "detail": str(audit_path),
        },
        {
            "resource": "second_pair_detector_module",
            "available": module_available,
            "detail": spd.DETECTOR_MODULE_PATH,
        },
    ]
    if not audit_artifact or not module_available:
        return build_artifact_from_measurements(
            audit_artifact=audit_artifact,
            ship_artifact={},
            module_available=module_available,
            code_surface_abstains=False,
            operating_envelope_docstring_updated=False,
            e2e_test_passed=False,
            adversarial_verify_clean=False,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            extra={"preconditions_checked": preconditions},
        )

    ship_artifact = spd.build_ship_artifact(root_path, tests_run=tests_run)
    return build_artifact_from_measurements(
        audit_artifact=audit_artifact,
        ship_artifact=ship_artifact,
        module_available=module_available,
        code_surface_abstains=code_surface_abstains(),
        operating_envelope_docstring_updated=operating_envelope_docstring_updated(),
        e2e_test_passed=bool(ship_artifact.get("e2e_test_passed")),
        adversarial_verify_clean=bool(audit_artifact.get("adversarial_verify_clean")),
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
        extra={
            "preconditions_checked": preconditions,
            "exp3705_audit_summary": _audit_summary(audit_artifact),
            "ship_artifact_summary": _ship_summary(ship_artifact),
        },
    )


def build_artifact_from_measurements(
    *,
    audit_artifact: Mapping[str, Any],
    ship_artifact: Mapping[str, Any],
    module_available: bool,
    code_surface_abstains: bool,
    operating_envelope_docstring_updated: bool,
    e2e_test_passed: bool,
    adversarial_verify_clean: bool,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the terminal artifact from measured reconciliation inputs."""

    start = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    action = reconciliation_action(audit_artifact, module_available=module_available)
    blocked = action == ACTION_BLOCKED
    heldout_code_auroc = _coerce_optional_float(audit_artifact.get("heldout_code_auroc"))
    shipped_code_auroc = (
        heldout_code_auroc if action == ACTION_RECALIBRATED else None
    )
    math_unchanged = False if blocked else math_operating_point_unchanged(ship_artifact)
    overclaim_removed = bool(
        not blocked
        and (
            action == ACTION_RECALIBRATED
            or (action == ACTION_NARROWED and code_surface_abstains)
        )
    )
    acceptance_passed = bool(overclaim_removed and math_unchanged and e2e_test_passed)
    verdict = VERDICT_BLOCKED
    if (
        action == ACTION_RECALIBRATED
        and acceptance_passed
        and operating_envelope_docstring_updated
        and adversarial_verify_clean
    ):
        verdict = VERDICT_RECALIBRATED
    elif (
        action == ACTION_NARROWED
        and acceptance_passed
        and operating_envelope_docstring_updated
        and adversarial_verify_clean
    ):
        verdict = VERDICT_NARROWED

    artifact: JsonDict = {
        "artifact": "experiment_3706_reconcile_shipped_detector_heldout",
        "schema": "carnot.reconcile_shipped_detector_heldout_3706.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "reconciliation_action": action,
        "shipped_code_operating_point_auroc": shipped_code_auroc,
        "math_operating_point_unchanged": bool(math_unchanged),
        "overclaim_removed": bool(overclaim_removed),
        "e2e_test_passed": bool(e2e_test_passed),
        "operating_envelope_docstring_updated": bool(
            operating_envelope_docstring_updated
        ),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _round(max(0.0, finished - start)),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "overclaim_removed == true AND "
                "math_operating_point_unchanged == true AND "
                "e2e_test_passed == true"
            ),
            "passed": acceptance_passed,
            "principle": (
                "The shipped detector is honest only if it no longer claims the "
                "inflated 1.0, the math point is preserved, and the E2E passes."
            ),
        },
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "heldout_calibration": dict(
            audit_artifact.get("heldout_calibration_brier_ece", {})
            if isinstance(audit_artifact.get("heldout_calibration_brier_ece"), Mapping)
            else {}
        ),
        "math_operating_point": _math_operating_point(ship_artifact),
        "code_surface_abstains": bool(code_surface_abstains),
    }
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def reconciliation_action(
    audit_artifact: Mapping[str, Any],
    *,
    module_available: bool,
) -> str:
    """Classify the reconciliation branch from Exp 3705's held-out verdict."""

    if not module_available or not audit_artifact:
        return ACTION_BLOCKED
    if audit_artifact.get("code_signal_survives_heldout") is True:
        return ACTION_RECALIBRATED
    if (
        audit_artifact.get("leak_detected") is True
        or audit_artifact.get("code_signal_survives_heldout") is False
    ):
        return ACTION_NARROWED
    return ACTION_BLOCKED


def math_operating_point_unchanged(ship_artifact: Mapping[str, Any]) -> bool:
    """Return true when the shipped math point remains AUROC 0.98/ECE 0.009."""

    auroc = _domain_value(ship_artifact, "fused_detector_auroc_per_domain", "math")
    calibration = _domain_mapping(ship_artifact, "calibration_brier_ece_per_domain", "math")
    if auroc is None or not calibration:
        return False
    return bool(
        round(float(auroc), 2) == 0.98
        and float(calibration.get("ece", math.inf)) <= 0.0095
    )


def detector_module_available() -> bool:
    """Return true when the shipped detector module exposes the runtime surface."""

    return bool(
        callable(getattr(spd, "score_candidates", None))
        and callable(getattr(spd, "build_ship_artifact", None))
    )


def code_surface_abstains() -> bool:
    """Return true when the shipped detector is narrowed to no-code-verdict."""

    scope = str(getattr(spd, "CODE_OPERATING_POINT_SCOPE", "")).lower()
    return bool(
        getattr(spd, "CODE_ABSTAIN_ON_CODE", False) is True
        and ("no_code_verdict" in scope or "no code verdict" in scope)
    )


def operating_envelope_docstring_updated() -> bool:
    """Return true when the module docstring and scope disclose code abstention."""

    doc = (inspect.getdoc(spd) or "").lower()
    scope = str(getattr(spd, "CODE_OPERATING_POINT_SCOPE", "")).lower()
    return bool(
        "math" in doc
        and ("abstain" in doc or "abstention" in doc)
        and "code_signal_survives_heldout=false" in scope
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3706 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("honest_verdict") not in TERMINAL_VERDICTS:
        raise ValueError("honest_verdict is not an accepted Exp 3706 terminal verdict")
    if artifact.get("reconciliation_action") not in {
        ACTION_RECALIBRATED,
        ACTION_NARROWED,
        ACTION_BLOCKED,
    }:
        raise ValueError("reconciliation_action is not accepted")
    for field in (
        "math_operating_point_unchanged",
        "overclaim_removed",
        "e2e_test_passed",
        "operating_envelope_docstring_updated",
        "adversarial_verify_clean",
    ):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare top-level bool")
    code_auroc = artifact.get("shipped_code_operating_point_auroc")
    if code_auroc is not None and not isinstance(code_auroc, (int, float)):
        raise ValueError("shipped_code_operating_point_auroc must be a number or null")
    if artifact.get("reconciliation_action") == ACTION_NARROWED and code_auroc is not None:
        raise ValueError("narrowed reconciliation must not claim a code AUROC")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3706 artifact fields."""

    payload = {
        "reconciliation_action": artifact.get("reconciliation_action"),
        "shipped_code_operating_point_auroc": artifact.get(
            "shipped_code_operating_point_auroc"
        ),
        "math_operating_point_unchanged": artifact.get("math_operating_point_unchanged"),
        "overclaim_removed": artifact.get("overclaim_removed"),
        "e2e_test_passed": artifact.get("e2e_test_passed"),
        "operating_envelope_docstring_updated": artifact.get(
            "operating_envelope_docstring_updated"
        ),
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
    """Build, adversarial-check, validate, and persist the Exp 3706 artifact."""

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
    """Persist a synthetic or pre-measured Exp 3706 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact_from_measurements(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_adversarial_verify_report(path: Path) -> JsonDict:
    """Run scripts/adversarial_verify.py against an artifact path."""

    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_3706", verifier_path)
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


def _audit_summary(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "code_signal_survives_heldout": artifact.get("code_signal_survives_heldout"),
        "leak_detected": artifact.get("leak_detected"),
        "heldout_code_auroc": artifact.get("heldout_code_auroc"),
        "heldout_calibration_brier_ece": artifact.get(
            "heldout_calibration_brier_ece"
        ),
    }


def _ship_summary(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "honest_verdict": artifact.get("honest_verdict"),
        "fused_detector_auroc_per_domain": artifact.get("fused_detector_auroc_per_domain"),
        "calibration_brier_ece_per_domain": artifact.get(
            "calibration_brier_ece_per_domain"
        ),
        "operating_points": artifact.get("operating_points"),
        "e2e_test_passed": artifact.get("e2e_test_passed"),
        "code_abstention": artifact.get("code_abstention"),
    }


def _math_operating_point(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "auroc": _domain_value(artifact, "fused_detector_auroc_per_domain", "math"),
        "calibration": _domain_mapping(
            artifact,
            "calibration_brier_ece_per_domain",
            "math",
        ),
        "operating_point": _domain_mapping(artifact, "operating_points", "math"),
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
    return _coerce_optional_float(value)


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


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float) -> float:
    if not math.isfinite(float(value)):
        return float(value)
    return round(float(value), 6)


__all__ = [
    "ACTION_BLOCKED",
    "ACTION_NARROWED",
    "ACTION_RECALIBRATED",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "build_artifact_from_measurements",
    "code_surface_abstains",
    "detector_module_available",
    "math_operating_point_unchanged",
    "operating_envelope_docstring_updated",
    "reconciliation_action",
    "validate_artifact",
    "write_artifact",
    "write_artifact_from_measurements",
]
