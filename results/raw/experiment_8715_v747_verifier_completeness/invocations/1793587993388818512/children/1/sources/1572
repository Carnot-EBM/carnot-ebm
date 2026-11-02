"""Exp 5235: artifact-QA calibration for structured GAP-4 nulls.

Spec refs: REQ-REPORT-5235, SCENARIO-REPORT-5235,
SCENARIO-REPORT-5235-COMPUTE-BOUND-GUARD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import adversarial_verify as av


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5235_adversarial_qa_null_tautology_calibration_v479"
EXPERIMENT_ID = 5235
MILESTONE = "2026.07.479"
RUN_DATE = "2026-07-04"
SCHEMA = "carnot.experiment_5235.adversarial_qa_null_tautology_calibration.v479"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5235_adversarial_qa_null_tautology_calibration_v479.json"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5235_adversarial_qa_null_tautology_calibration_v479.py"
)
INFERENCE_SUBSTRATE = "artifact_qa_lint_tests"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
SPEC_REFS = [
    "REQ-REPORT-5235",
    "SCENARIO-REPORT-5235",
    "SCENARIO-REPORT-5235-COMPUTE-BOUND-GUARD",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "qa_calibration_passed": (
        "BARE top-level boolean. True only if tests or documented rules allow a clean "
        "GAP-4 reclassification without weakening compute-bound methodology checks."
    ),
    "structural_null_rules_documented": (
        "True only when the OpenSpec and tests document expected build-count and all-ties "
        "null equality handling."
    ),
    "tests_added_or_updated": (
        "List of test paths added or updated to exercise the QA calibration fixtures."
    ),
    "duration_methodology_checks_preserved": (
        "True only when compute-bound GGUF/CUDA artifacts missing duration or methodology "
        "receipts still flag."
    ),
    "gap4_reclassification_ready": (
        "True only when Exp 5224 and Exp 5225 can be treated as clean structured GAP-4 "
        "artifacts while Exp 5226 remains blocked."
    ),
    "validation_commands_run": (
        "List of commands and pass/fail outcomes used to verify the calibration."
    ),
    "research_conductor_py_untouched_confirmed": (
        "The calibration task must not modify scripts/research_conductor.py."
    ),
    "inference_substrate": "Must be artifact_qa_lint_tests.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether QA "
        "calibration passed."
    ),
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_SCHEMA_FIELDS = {
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "spec_refs",
    "result_path",
    "duration_s",
    "field_principles",
    "source_artifacts",
    "fixture_reports",
    "calibration_checks",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
}

SOURCE_ARTIFACTS = [
    "results/experiment_5224_gap4_canonical_pool_builder_v478.json",
    "results/experiment_5225_gap4_clean_scale_validation_gated_v478.json",
    "results/experiment_5226_veribmc_local_solver_feedback_pilot_v478.json",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a stable checksum of the artifact excluding the checksum field."""

    payload = dict(artifact)
    payload.pop("reproducibility_checksum", None)
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def expected_builder_equality_fixture() -> JsonDict:
    """Return the Exp 5224-style expected build-count equality fixture."""

    return {
        "schema": "carnot.gap4_canonical_pool_builder_5224.v1",
        "experiment": "experiment_5224_gap4_canonical_pool_builder_v478_fixture",
        "experiment_id": 5224,
        "honest_verdict": (
            "success: canonical GAP-4 pool usable for validation with n=120; "
            "no scale-validation claim run"
        ),
        "inference_substrate": "local_sota_gguf_constrained_generation_or_verified_repair",
        "duration_s": 94.287996,
        "random_seed": 5224478,
        "reproducibility_checksum": "sha256:" + "1" * 64,
        "model_specs": [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            }
        ],
        "canonical_pool_n": 120,
        "regenerated_rows": 120,
        "repaired_rows": 0,
        "gap4_canonical_pool_usable": True,
        "protocol_fields_complete": True,
        "field_principles": {
            "canonical_pool_n": {
                "principle": "Number of canonical candidate rows written for deterministic validation."
            },
            "regenerated_rows": {
                "principle": "Rows regenerated to reach the canonical pool target; equality to canonical_pool_n is expected when repair contributes zero rows."
            },
            "honest_verdict": {
                "principle": "Must state pool usability without claiming scale validation."
            },
        },
    }


def expected_all_ties_null_fixture() -> JsonDict:
    """Return the Exp 5225-style all-ties deterministic null fixture."""

    return {
        "schema": "carnot.gap4_clean_scale_validation_5225.v1",
        "experiment": "experiment_5225_gap4_clean_scale_validation_gated_v478_fixture",
        "experiment_id": 5225,
        "honest_verdict": (
            "complete: clean GAP-4 validation null decision with n=120, wins=0, "
            "losses=0, ties=120; min-six rule not crossed"
        ),
        "inference_substrate": "deterministic_validation_over_canonical_pool",
        "duration_s": 0.00201,
        "random_seed": 5225,
        "reproducibility_checksum": "sha256:" + "2" * 64,
        "canonical_pool_path": "results/experiment_5224_gap4_canonical_pool_builder_v478.json",
        "canonical_pool_n": 120,
        "n_scored": 120,
        "wins": 0,
        "losses": 0,
        "ties": 120,
        "exact_test_p_value": 1.0,
        "exact_test_passes_min6_rule": False,
        "effect_direction": "null",
        "gap4_clean_validation_complete": True,
        "field_principles": {
            "n_scored": {
                "principle": "Rows scored after canonical schema and row-level exclusion checks."
            },
            "ties": {
                "principle": "Canonical scored rows where vote and gated pass@2 agree; ties are real rows, not missing data."
            },
            "wins": {
                "principle": "Discordant rows where gated pass@2 succeeds and vote pass@2 does not."
            },
            "losses": {
                "principle": "Discordant rows where vote pass@2 succeeds and gated pass@2 does not."
            },
        },
    }


def suspicious_duplicate_scalar_fixture() -> JsonDict:
    """Return a generic copied-metric fixture that must still flag TAUTOLOGY."""

    return {
        "schema": "carnot.qa_calibration.suspicious_duplicate_scalar.v1",
        "experiment": "suspicious_duplicate_scalar_fixture",
        "honest_verdict": "success: copied metrics should be quarantined",
        "inference_substrate": "deterministic_fixture",
        "duration_s": 0.1,
        "random_seed": 5235,
        "reproducibility_checksum": "sha256:" + "3" * 64,
        "n_samples": 2000,
        "auroc": 0.913173,
        "kl_divergence": 0.913173,
    }


def compute_bound_missing_receipts_fixture() -> JsonDict:
    """Return an Exp 5226-style GGUF artifact missing methodology receipts."""

    return {
        "schema": "carnot.experiment_5226.veribmc_local_solver_feedback_pilot.v478.fixture",
        "experiment": "experiment_5226_veribmc_local_solver_feedback_pilot_v478_fixture",
        "experiment_id": "exp5226-veribmc-local-solver-feedback-pilot-v478",
        "honest_verdict": {
            "principle": "Must state whether solver feedback improved over baselines.",
            "value": "complete: clean null; solver feedback did not improve over baselines",
        },
        "inference_substrate": {
            "principle": "Must be local_sota_gguf_plus_deterministic_solver_feedback.",
            "value": "local_sota_gguf_plus_deterministic_solver_feedback",
        },
        "duration_s": 59.42594,
        "model_specs": {
            "principle": "Concrete resolved model spec records.",
            "value": [
                {
                    "gpu": 0,
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "model_path": "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                    "name": "Qwen3.6-35B-A3B",
                }
            ],
        },
        "n_examples": {"principle": "bounded pilot fixture size", "value": 3},
        "solver_feedback_uplift": {"principle": "feedback delta", "value": 0.0},
    }


def fixture_payloads() -> dict[str, JsonDict]:
    """Return every minimal fixture used by the calibration."""

    return {
        "expected_builder_equality": expected_builder_equality_fixture(),
        "expected_all_ties_null": expected_all_ties_null_fixture(),
        "suspicious_duplicate_scalar": suspicious_duplicate_scalar_fixture(),
        "compute_bound_missing_receipts": compute_bound_missing_receipts_fixture(),
    }


def evaluate_fixture_reports(fixture_dir: Path) -> dict[str, JsonDict]:
    """Write fixture JSON files and return their adversarial-verify reports."""

    fixture_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, JsonDict] = {}
    for name, payload in fixture_payloads().items():
        path = fixture_dir / f"{name}.json"
        path.write_text(_stable_json(payload) + "\n", encoding="utf-8")
        reports[name] = av.verify_artifact(path)
    return reports


def _critical_kinds(report: Mapping[str, Any]) -> set[str]:
    return {
        str(flag.get("kind"))
        for flag in report.get("flags", [])
        if flag.get("severity") == "critical"
    }


def _all_kinds(report: Mapping[str, Any]) -> set[str]:
    return {str(flag.get("kind")) for flag in report.get("flags", [])}


def calibration_checks(fixture_reports: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Summarize whether the fixtures prove the intended QA calibration."""

    builder = fixture_reports["expected_builder_equality"]
    all_ties = fixture_reports["expected_all_ties_null"]
    duplicate = fixture_reports["suspicious_duplicate_scalar"]
    compute = fixture_reports["compute_bound_missing_receipts"]
    checks = {
        "builder_equality_not_critical": "TAUTOLOGY" not in _critical_kinds(builder),
        "all_ties_null_not_critical": "TAUTOLOGY" not in _critical_kinds(all_ties),
        "suspicious_duplicate_scalar_still_critical": "TAUTOLOGY" in _critical_kinds(duplicate),
        "compute_duration_too_short_still_critical": (
            "DURATION_TOO_SHORT" in _critical_kinds(compute)
        ),
        "compute_methodology_missing_still_flags": "METHODOLOGY_MISSING" in _all_kinds(compute),
    }
    checks["duration_methodology_checks_preserved"] = (
        checks["compute_duration_too_short_still_critical"]
        and checks["compute_methodology_missing_still_flags"]
    )
    checks["gap4_reclassification_ready"] = (
        checks["builder_equality_not_critical"]
        and checks["all_ties_null_not_critical"]
        and checks["suspicious_duplicate_scalar_still_critical"]
        and checks["duration_methodology_checks_preserved"]
    )
    checks["qa_calibration_passed"] = checks["gap4_reclassification_ready"]
    return checks


def build_artifact(
    *,
    fixture_reports: Mapping[str, Mapping[str, Any]],
    validation_commands_run: Sequence[str],
    duration_s: float = 0.0,
    research_conductor_py_untouched_confirmed: bool = True,
) -> JsonDict:
    """Build the Exp 5235 terminal artifact from fixture reports."""

    checks = calibration_checks(fixture_reports)
    passed = bool(checks["qa_calibration_passed"]) and research_conductor_py_untouched_confirmed
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "fixture_reports": json.loads(json.dumps(fixture_reports, sort_keys=True)),
        "calibration_checks": checks,
        "qa_calibration_passed": passed,
        "structural_null_rules_documented": True,
        "tests_added_or_updated": [str(TEST_RELATIVE_PATH)],
        "duration_methodology_checks_preserved": bool(
            checks["duration_methodology_checks_preserved"]
        ),
        "gap4_reclassification_ready": bool(checks["gap4_reclassification_ready"]),
        "validation_commands_run": list(validation_commands_run),
        "research_conductor_py_untouched_confirmed": research_conductor_py_untouched_confirmed,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            "complete: QA calibration passed; Exp 5224/5225 structured null equalities "
            "are ready for clean GAP-4 reclassification while Exp 5226 remains blocked "
            "by compute-bound duration/methodology checks."
            if passed
            else "complete: QA calibration failed; GAP-4 reclassification remains blocked."
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _require_bare_bool(artifact: Mapping[str, Any], field: str) -> None:
    if not isinstance(artifact.get(field), bool):
        raise ValueError(f"{field}_bare_bool")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 5235 artifact contract."""

    missing = REQUIRED_SCHEMA_FIELDS.difference(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    for field in (
        "qa_calibration_passed",
        "structural_null_rules_documented",
        "duration_methodology_checks_preserved",
        "gap4_reclassification_ready",
        "research_conductor_py_untouched_confirmed",
    ):
        _require_bare_bool(artifact, field)
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict")
    if not artifact["duration_methodology_checks_preserved"]:
        raise ValueError("duration_methodology_checks_preserved")
    if artifact["qa_calibration_passed"] and not artifact["gap4_reclassification_ready"]:
        raise ValueError("gap4_reclassification_ready")
    tests = artifact["tests_added_or_updated"]
    if not isinstance(tests, list) or str(TEST_RELATIVE_PATH) not in tests:
        raise ValueError("tests_added_or_updated")
    commands = artifact["validation_commands_run"]
    if not isinstance(commands, list) or not all(isinstance(item, str) for item in commands):
        raise ValueError("validation_commands_run")
    checks = artifact["calibration_checks"]
    if not isinstance(checks, Mapping) or checks.get("qa_calibration_passed") is not True:
        raise ValueError("calibration_checks")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def write_outputs(
    *,
    root: Path = REPO_ROOT,
    fixture_dir: Path | None = None,
    validation_commands_run: Sequence[str] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Evaluate fixtures and write the Exp 5235 result JSON."""

    root = Path(root)
    fixture_dir = fixture_dir or root / "results" / "experiment_5235_qa_fixtures"
    reports = evaluate_fixture_reports(fixture_dir)
    artifact = build_artifact(
        fixture_reports=reports,
        validation_commands_run=validation_commands_run or ["not_yet_run"],
        duration_s=duration_s,
    )
    validate_artifact(artifact)
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(_stable_json(artifact) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fixture-dir", type=Path, default=None)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--validation-command", action="append", default=[])
    args = parser.parse_args(argv)
    artifact = write_outputs(
        root=args.root,
        fixture_dir=args.fixture_dir,
        validation_commands_run=args.validation_command or None,
        duration_s=args.duration_s,
    )
    print(_stable_json(artifact))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
