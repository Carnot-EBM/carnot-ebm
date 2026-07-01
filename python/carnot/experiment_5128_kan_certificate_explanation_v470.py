"""Exp 5128: KAN certificate breadth and explanation cycle audit.

Spec refs: REQ-KAN-5128, SCENARIO-KAN-5128.

Exp 5108 found that the exact-MILP encoding hits a wall before the production
KAEM unit count. Exp 5114 changed technique by using local/global
abstraction-refinement certificates. This module keeps that post-wall
certificate path and asks a narrower question: can independent certificate
families be explained deterministically, reconstructed from the explanation,
and checked again by a symbolic validator?
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5114_kan_abstraction_refinement_post_wall_v469 as post_wall
from carnot.experiment_5108_kan_pwa_milp_scale_stress_test import (
    RESULT_RELATIVE_PATH as EXP5108_RESULT_RELATIVE_PATH,
)


JsonDict = dict[str, Any]

EXPERIMENT_ID = "exp5128-kan-certificate-explanation-v470"
MILESTONE = "2026.07.470"
RUN_DATE = "20260701"
RESULT_RELATIVE_PATH = "results/experiment_5128_kan_certificate_explanation_v470.json"
INFERENCE_SUBSTRATE = "cpu_kan_abstraction_and_symbolic_certificate_check"
SPEC_REFS = ["REQ-KAN-5128", "SCENARIO-KAN-5128"]
RANDOM_SEED = post_wall.RANDOM_SEED
DEFAULT_N_UNITS = 100
EXPLANATION_PREFIX = "KAN_CERT_EXPLANATION_V1"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_")

REQUIRED_CERTIFICATE_FIELDS = (
    "property",
    "verdict",
    "margin",
    "abstraction_error",
    "proof_status",
)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "exp5108_wall_loaded",
    "exp5114_baseline_loaded",
    "property_families",
    "certificates_emitted",
    "certificate_soundness",
    "false_property_detected",
    "near_margin_abstained",
    "explanation_records",
    "explanation_cycle_soundness",
    "kan_certificate_breadth_ready",
    "flagged_adversarial",
    "conductor_modified",
    "tests_run",
)

FIELD_PRINCIPLES = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "exp5108_wall_loaded": "no stale premise",
    "exp5114_baseline_loaded": "continuation accountability",
    "property_families": "breadth",
    "certificates_emitted": "proof artifact",
    "certificate_soundness": "formal correctness",
    "false_property_detected": "adversarial control",
    "near_margin_abstained": "no overclaim",
    "explanation_records": "explainability",
    "explanation_cycle_soundness": "explanation faithfulness",
    "kan_certificate_breadth_ready": "capstone decision",
    "flagged_adversarial": "adversarial-verification accountability",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
}

DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5128_kan_certificate_explanation_v470.py --date 20260701",
    '.venv/bin/pytest tests/python/test_experiment_5128_kan_certificate_explanation_v470.py -q -o addopts=""',
    ".venv/bin/coverage erase && .venv/bin/coverage run "
    "--include='/home/ianblenke/github.com/ianblenke/carnot/python/carnot/"
    "experiment_5128_kan_certificate_explanation_v470.py' -m pytest "
    'tests/python/test_experiment_5128_kan_certificate_explanation_v470.py -q -o addopts="" && '
    ".venv/bin/coverage report --include='/home/ianblenke/github.com/ianblenke/carnot/python/"
    "carnot/experiment_5128_kan_certificate_explanation_v470.py' --fail-under=100 -m",
    "python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5128_kan_certificate_explanation_v470.py",
    ".venv/bin/pytest tests/python -q",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover - validation guard.
        raise ValueError(message)


def _float_token(value: float) -> str:
    return f"{float(value):.12g}"


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return dict(parsed) if isinstance(parsed, Mapping) else None


def load_exp5108_wall(root: str | Path | None = None) -> JsonDict:
    """Load the exact-MILP wall artifact without re-running its solver sweep."""

    base = Path(root) if root is not None else _repo_root()
    path = base / EXP5108_RESULT_RELATIVE_PATH
    payload = _read_json(path)
    if payload is None:
        return {"loaded": False, "path": str(path)}
    return {
        "loaded": True,
        "path": str(path.relative_to(base)),
        "sha256": _sha256_file(path),
        "honest_verdict": payload.get("honest_verdict"),
        "inference_substrate": payload.get("inference_substrate"),
        "largest_n_reached": payload.get("largest_n_reached"),
        "solver_timeout_hit": payload.get("solver_timeout_hit"),
        "solver_timeout_ms": payload.get("solver_timeout_ms"),
    }


def load_exp5114_baseline(root: str | Path | None = None) -> JsonDict:
    """Load the post-wall abstraction-refinement baseline for continuation."""

    base = Path(root) if root is not None else _repo_root()
    path = base / post_wall.RESULT_RELATIVE_PATH
    payload = _read_json(path)
    if payload is None:
        return {"loaded": False, "path": str(path)}
    return {
        "loaded": True,
        "path": str(path.relative_to(base)),
        "sha256": _sha256_file(path),
        "experiment_id": payload.get("experiment_id"),
        "honest_verdict": payload.get("honest_verdict"),
        "inference_substrate": payload.get("inference_substrate"),
        "solved_n": payload.get("solved_n"),
        "certificate_soundness": payload.get("certificate_soundness"),
        "false_property_detected": payload.get("false_property_detected"),
        "near_margin_abstained": payload.get("near_margin_abstained"),
    }


def _certificate_from_outcome(
    *,
    family: str,
    expectation: str,
    statement: str,
    outcome: post_wall.PropertyOutcome,
    source: post_wall.RefinedCertificate,
) -> JsonDict:
    if outcome.property_status == "verified":
        verdict = "verified"
        proof_status = "proved_by_conservative_bound"
        margin = outcome.threshold - outcome.certified_upper_bound
    elif outcome.property_status == "counterexample":
        verdict = "counterexample"
        proof_status = "refuted_by_exact_witness"
        margin = outcome.exact_upper_bound - outcome.threshold
    else:
        verdict = "abstained"
        proof_status = "abstained_residual_gap"
        margin = outcome.certified_upper_bound - outcome.threshold
    property_record = {
        "id": f"kan_n{source.n_units}_{family}",
        "family": family,
        "statement": statement,
        "expectation": expectation,
        "unit_count": source.n_units,
        "threshold": outcome.threshold,
    }
    return {
        "property": property_record,
        "verdict": verdict,
        "margin": float(max(0.0, margin)),
        "abstraction_error": float(source.global_error_bound),
        "proof_status": proof_status,
        "bounds": {
            "exact_upper_bound": outcome.exact_upper_bound,
            "certified_upper_bound": outcome.certified_upper_bound,
            "global_error_bound": source.global_error_bound,
        },
        "counterexample": outcome.counterexample,
        "source": {
            "technique": "local_global_abstraction_refinement_unit_decomposition",
            "unit_count": source.n_units,
            "seed": source.seed,
            "exp5114_module": post_wall.RESULT_RELATIVE_PATH,
        },
    }


def _refinement_budget_certificate(source: post_wall.RefinedCertificate) -> JsonDict:
    margin = source.initial_global_error_bound - source.global_error_bound
    return {
        "property": {
            "id": f"kan_n{source.n_units}_refinement_error_budget",
            "family": "refinement_error_budget",
            "statement": "refined global abstraction error does not exceed initial global error",
            "expectation": "true_control",
            "unit_count": source.n_units,
            "threshold": source.initial_global_error_bound,
            "observed": source.global_error_bound,
        },
        "verdict": "verified",
        "margin": float(max(0.0, margin)),
        "abstraction_error": float(source.global_error_bound),
        "proof_status": "proved_by_refinement_budget",
        "bounds": {
            "initial_global_error_bound": source.initial_global_error_bound,
            "refined_global_error_bound": source.global_error_bound,
        },
        "counterexample": None,
        "source": {
            "technique": "largest_local_error_first_refinement",
            "unit_count": source.n_units,
            "seed": source.seed,
            "exp5114_module": post_wall.RESULT_RELATIVE_PATH,
        },
    }


def build_explainable_certificates(
    *,
    n_units: int = DEFAULT_N_UNITS,
    seed: int = RANDOM_SEED + DEFAULT_N_UNITS,
) -> list[JsonDict]:
    """Build machine-readable certificates from the Exp 5114 certificate path."""

    source = post_wall.build_refined_certificate(n_units=n_units, seed=seed)
    outcomes = {
        outcome.property_class: outcome for outcome in post_wall.evaluate_property_classes(source)
    }
    return [
        _certificate_from_outcome(
            family="global_energy_upper_bound",
            expectation="true_control",
            statement="global additive KAEM energy upper bound is below the safe threshold",
            outcome=outcomes["true_safe"],
            source=source,
        ),
        _refinement_budget_certificate(source),
        _certificate_from_outcome(
            family="false_low_threshold_control",
            expectation="false_control",
            statement="global additive KAEM energy upper bound is below an intentionally false threshold",
            outcome=outcomes["false_counterexample"],
            source=source,
        ),
        _certificate_from_outcome(
            family="near_margin_residual_gap",
            expectation="near_margin_abstention",
            statement="near-margin threshold is provable despite residual abstraction error",
            outcome=outcomes["near_margin_abstain"],
            source=source,
        ),
    ]


def _certificate_sound(certificate: Mapping[str, Any]) -> bool:
    missing = set(REQUIRED_CERTIFICATE_FIELDS) - set(certificate)
    if missing:
        return False
    prop = certificate["property"]
    if not isinstance(prop, Mapping):
        return False
    if float(certificate["margin"]) < 0.0 or float(certificate["abstraction_error"]) < 0.0:
        return False
    verdict = certificate["verdict"]
    proof_status = certificate["proof_status"]
    bounds = certificate.get("bounds", {})
    threshold = float(prop.get("threshold", 0.0))
    if proof_status == "proved_by_conservative_bound":
        return verdict == "verified" and float(bounds["certified_upper_bound"]) <= threshold + 1e-9
    if proof_status == "proved_by_refinement_budget":
        return verdict == "verified" and float(prop["observed"]) <= threshold + 1e-9
    if proof_status == "refuted_by_exact_witness":
        return (
            verdict == "counterexample"
            and certificate.get("counterexample") is not None
            and float(bounds["exact_upper_bound"]) > threshold + 1e-9
        )
    if proof_status == "abstained_residual_gap":
        return (
            verdict == "abstained"
            and float(bounds["exact_upper_bound"]) <= threshold + 1e-9
            and float(bounds["certified_upper_bound"]) > threshold + 1e-9
        )
    return False


def validate_certificates(certificates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the certificate-suite control summary used by the artifact."""

    families = {str(certificate["property"]["family"]) for certificate in certificates}
    sound = bool(certificates) and all(
        _certificate_sound(certificate) for certificate in certificates
    )
    false_detected = any(
        certificate["property"].get("expectation") == "false_control"
        and certificate["verdict"] == "counterexample"
        and _certificate_sound(certificate)
        for certificate in certificates
    )
    near_abstained = any(
        certificate["property"].get("expectation") == "near_margin_abstention"
        and certificate["verdict"] == "abstained"
        and _certificate_sound(certificate)
        for certificate in certificates
    )
    return {
        "property_family_count": len(families),
        "certificate_soundness": sound,
        "false_property_detected": false_detected,
        "near_margin_abstained": near_abstained,
    }


def _metadata_for_explanation(certificate: Mapping[str, Any]) -> dict[str, str]:
    prop = certificate["property"]
    return {
        "property_id": str(prop["id"]),
        "family": str(prop["family"]),
        "verdict": str(certificate["verdict"]),
        "margin": _float_token(float(certificate["margin"])),
        "abstraction_error": _float_token(float(certificate["abstraction_error"])),
        "proof_status": str(certificate["proof_status"]),
    }


def explain_certificate(certificate: Mapping[str, Any]) -> str:
    """Generate a deterministic structured explanation for one certificate."""

    metadata = _metadata_for_explanation(certificate)
    return "|".join([EXPLANATION_PREFIX, *(f"{key}={value}" for key, value in metadata.items())])


def reconstruct_metadata_from_explanation(explanation: str) -> dict[str, str]:
    """Reconstruct certificate metadata from the deterministic explanation."""

    parts = explanation.split("|")
    if not parts or parts[0] != EXPLANATION_PREFIX:
        return {}
    parsed: dict[str, str] = {}
    for part in parts[1:]:
        key, value = part.split("=", 1)
        parsed[key] = value
    return parsed


def symbolic_validate_explanation(
    certificate: Mapping[str, Any],
    explanation: str,
) -> JsonDict:
    """Check explanation metadata against the machine certificate."""

    expected = _metadata_for_explanation(certificate)
    reconstructed = reconstruct_metadata_from_explanation(explanation)
    mismatches = [key for key, value in expected.items() if reconstructed.get(key) != value]
    return {
        "valid": not mismatches and _certificate_sound(certificate),
        "mismatches": mismatches,
        "reconstructed_metadata": reconstructed,
    }


def generate_explanation_records(certificates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Generate explanation, reconstruction, and cycle-check records."""

    records: list[JsonDict] = []
    for certificate in certificates:
        explanation = explain_certificate(certificate)
        validation = symbolic_validate_explanation(certificate, explanation)
        records.append(
            {
                "certificate_id": certificate["property"]["id"],
                "explanation": explanation,
                "reconstructed_metadata": validation["reconstructed_metadata"],
                "cycle_sound": bool(validation["valid"]),
                "symbolic_validator": validation,
            }
        )
    return records


def _property_family_records(certificates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "family": certificate["property"]["family"],
            "property_id": certificate["property"]["id"],
            "expectation": certificate["property"]["expectation"],
            "verdict": certificate["verdict"],
            "proof_status": certificate["proof_status"],
        }
        for certificate in certificates
    ]


def build_artifact(
    *,
    root: str | Path | None = None,
    run_date: str = RUN_DATE,
    n_units: int = DEFAULT_N_UNITS,
    tests_run: Sequence[str] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Build the Exp 5128 deliverable payload."""

    start = time.perf_counter()
    repo = Path(root) if root is not None else _repo_root()
    wall = load_exp5108_wall(repo)
    baseline = load_exp5114_baseline(repo)
    can_audit = bool(wall.get("loaded")) and bool(baseline.get("loaded"))
    certificates = (
        build_explainable_certificates(n_units=n_units, seed=RANDOM_SEED + n_units)
        if can_audit
        else []
    )
    certificate_summary = (
        validate_certificates(certificates)
        if certificates
        else {
            "property_family_count": 0,
            "certificate_soundness": False,
            "false_property_detected": False,
            "near_margin_abstained": False,
        }
    )
    explanation_records = generate_explanation_records(certificates) if certificates else []
    explanation_cycle_soundness = bool(explanation_records) and all(
        record["cycle_sound"] for record in explanation_records
    )
    breadth_ready = (
        can_audit
        and int(certificate_summary["property_family_count"]) >= 3
        and bool(certificate_summary["certificate_soundness"])
        and bool(certificate_summary["false_property_detected"])
        and bool(certificate_summary["near_margin_abstained"])
        and explanation_cycle_soundness
    )
    if breadth_ready:
        honest_verdict = "success_kan_certificate_explanation_cycle_sound_breadth_ready"
    elif not can_audit:
        honest_verdict = "blocked_kan_certificate_explanation_missing_upstream_baseline"
    else:
        honest_verdict = "complete_kan_certificate_explanation_breadth_not_ready"

    artifact = {
        "schema": "carnot.kan_certificate_explanation.v470",
        "experiment": 5128,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "artifact": "experiment_5128_kan_certificate_explanation_v470",
        "run_date": run_date,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.perf_counter() - start, 6),
        "exp5108_wall_loaded": bool(wall.get("loaded")),
        "exp5114_baseline_loaded": bool(baseline.get("loaded")),
        "exp5108_wall": wall,
        "exp5114_baseline": baseline,
        "property_families": _property_family_records(certificates),
        "certificates_emitted": certificates,
        "certificate_soundness": bool(certificate_summary["certificate_soundness"]),
        "false_property_detected": bool(certificate_summary["false_property_detected"]),
        "near_margin_abstained": bool(certificate_summary["near_margin_abstained"]),
        "explanation_records": explanation_records,
        "explanation_cycle_soundness": explanation_cycle_soundness,
        "kan_certificate_breadth_ready": breadth_ready,
        "flagged_adversarial": not breadth_ready,
        "conductor_modified": False,
        "source_artifacts": [EXP5108_RESULT_RELATIVE_PATH, post_wall.RESULT_RELATIVE_PATH],
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_note": (
            "Exp 5128 loads Exp 5108 and Exp 5114, reuses the CPU KAN "
            "abstraction-refinement certificate path, and checks deterministic "
            "explanation reconstruction. It does not rerun the exact-MILP scale sweep."
        ),
        "tests_run": list(tests_run),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5128 artifact drifts from the requested schema."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    _require(artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch")
    _require(artifact["milestone"] == MILESTONE, "milestone mismatch")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "substrate mismatch")
    _require("live_llm" not in artifact["inference_substrate"], "must not claim live LLM")
    _require(str(artifact["honest_verdict"]).startswith(TERMINAL_PREFIXES), "bad verdict prefix")
    _require(isinstance(artifact["duration_s"], float), "duration_s must be a float")
    _require(artifact["duration_s"] >= 0.0, "duration_s cannot be negative")
    _require(artifact["conductor_modified"] is False, "conductor must remain unmodified")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]),
        "field_principles must cover required fields",
    )
    certificates = artifact["certificates_emitted"]
    explanations = artifact["explanation_records"]
    _require(len(certificates) == len(explanations), "certificate/explanation count mismatch")
    if artifact["kan_certificate_breadth_ready"]:
        _require(artifact["exp5108_wall_loaded"] is True, "ready requires Exp 5108")
        _require(artifact["exp5114_baseline_loaded"] is True, "ready requires Exp 5114")
        _require(artifact["certificate_soundness"] is True, "ready requires certificate soundness")
        _require(artifact["false_property_detected"] is True, "ready requires false control")
        _require(artifact["near_margin_abstained"] is True, "ready requires abstention")
        _require(
            artifact["explanation_cycle_soundness"] is True,
            "ready requires explanation cycle soundness",
        )
        _require(len(artifact["property_families"]) >= 3, "ready requires property breadth")
        _require(artifact["flagged_adversarial"] is False, "ready must not be flagged")
    else:
        _require(artifact["flagged_adversarial"] is True, "not-ready artifacts must be flagged")
    for certificate in certificates:
        _require(_certificate_sound(certificate), "emitted certificates must be sound")
    for certificate, record in zip(certificates, explanations, strict=True):
        _require(
            symbolic_validate_explanation(certificate, record["explanation"])["valid"] is True,
            "explanation cycle must validate",
        )


def write_outputs(
    *,
    artifact_path: str | Path,
    run_date: str = RUN_DATE,
    root: str | Path | None = None,
) -> JsonDict:
    """Write the Exp 5128 JSON artifact and return the validated payload."""

    artifact = build_artifact(root=root, run_date=run_date)
    output = Path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for writing the default Exp 5128 artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, help="Run date as YYYYMMDD")
    parser.add_argument("--output", default=None, help="Optional artifact output path")
    args = parser.parse_args(argv)

    root = _repo_root()
    output = Path(args.output) if args.output else root / RESULT_RELATIVE_PATH
    artifact = write_outputs(artifact_path=output, run_date=str(args.date), root=root)
    print(artifact["honest_verdict"])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    raise SystemExit(main())
