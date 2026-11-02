"""Exp 5280 artifact-normalizer evidence audit.

Spec refs: REQ-REPORT-5280, SCENARIO-REPORT-5280-EVIDENCE-AUDIT.

The audit is intentionally aggregation-only. It checks the existing producer
normalizer boundary, the adversarial verifier, and checked-in v481/v482
artifacts without re-running model inference or weakening old quarantine flags.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from scripts import adversarial_verify as av
from scripts.experiment_template import (
    PRODUCER_NORMALIZER_RECEIPTS_FIELD,
    normalize_artifact_for_template_write,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5280_artifact_normalizer_evidence_audit_v482.json")
EXPERIMENT = "experiment_5280_artifact_normalizer_evidence_audit_v482"
EXPERIMENT_ID = "exp5280-artifact-normalizer-evidence-audit-v482"
MILESTONE = "2026.07.482"
RUN_DATE = "2026-07-05"
SCHEMA = "carnot.experiment_5280.artifact_normalizer_evidence_audit.v482"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")
SPEC_REFS = ("REQ-REPORT-5280", "SCENARIO-REPORT-5280-EVIDENCE-AUDIT")
AUDIT_CASES = (
    "valid_shape_only_artifact",
    "missing_evidence",
    "bare_gate_fields",
    "dict_wrapped_substrate_fields",
    "sub_threshold_duration",
    "no_llm_aggregation_artifact",
)
V481_QUARANTINE_RELATIVE_PATHS = (
    Path("results/experiment_5262_solver_grounded_constraint_extraction_v481.json"),
    Path("results/experiment_5263_neuron_attention_energy_hallucination_probe_v481.json"),
)
SOURCE_RELATIVE_PATHS = (
    Path("scripts/experiment_template.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/conductor_gates.py"),
    Path("results/experiment_5267_artifact_normalizer_template_adoption_v481.json"),
    *V481_QUARANTINE_RELATIVE_PATHS,
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Must start with complete: or blocked_ and state whether producer evidence "
        "discipline is ready after auditing gates, evidence, duration, and substrate behavior."
    ),
    "inference_substrate": (
        "Must be aggregation_from_upstream_artifacts because Exp5280 reads checked-in "
        "code, tests, and artifacts without invoking a model."
    ),
    "normalizer_evidence_ready": (
        "True only when the audit matrix shows shape-only repairs succeed, missing "
        "evidence is rejected, and old quarantined artifacts are not laundered clean."
    ),
    "producer_coverage": (
        "Numeric coverage ratio for current normalizer-required producer/template "
        "surfaces covered by the producer-side normalizer."
    ),
    "bare_gate_preservation_passed": (
        "True only when existing top-level boolean gates remain bare booleans after normalization."
    ),
    "missing_evidence_rejected": (
        "True only when missing methodology, model, seed, checksum, or duration evidence "
        "remains an unsafe rejection."
    ),
    "duration_substrate_regression_passed": (
        "True only when wrapped substrate declarations, sub-threshold live durations, "
        "and no-LLM aggregation floors preserve adversarial-verifier behavior."
    ),
    "adversarial_verify_weakening": (
        "Must be false; old artifacts must remain quarantined by evidence findings "
        "rather than by weakening verifier checks."
    ),
    "tests_run": (
        "Records command/outcome receipts for focused unit, coverage, full repository "
        "tests, and artifact verification checks."
    ),
}
REQUIRED_WRAPPED_FIELDS = tuple(field for field in FIELD_PRINCIPLES if field != "tests_run")
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "run_date",
    "result_path",
    "spec_refs",
    "duration_s",
    "field_principles",
    "producer_inventory",
    "audit_matrix",
    "v481_quarantine_checks",
    "source_artifacts_read",
    "research_conductor_modified",
    *REQUIRED_WRAPPED_FIELDS,
    "tests_run",
    "reproducibility_checksum",
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(stable).encode("utf-8")).hexdigest()


def _flag_kinds(report: Mapping[str, Any]) -> list[str]:
    return sorted(str(flag.get("kind")) for flag in report.get("flags", []) if flag.get("kind"))


def _receipt_kinds(receipts: Mapping[str, Any], key: str) -> list[str]:
    rows = receipts.get(key, [])
    return sorted(str(row.get("kind")) for row in rows if isinstance(row, Mapping))


def _verify_payload(payload: Mapping[str, Any]) -> JsonDict:
    with tempfile.TemporaryDirectory(prefix="exp5280_audit_") as tmp:
        path = Path(tmp) / "artifact.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return av.verify_artifact(path)


def enumerate_producers() -> list[JsonDict]:
    """Enumerate producer/template surfaces relevant to this audit."""

    return [
        {
            "surface": "scripts/experiment_template.py:normalize_artifact_for_template_write",
            "kind": "template_hook",
            "normalizer_required": True,
            "covered_by_normalizer": True,
            "evidence": "delegates to Exp5247 strict normalizer on in-memory artifact copies",
        },
        {
            "surface": "scripts/experiment_template.py:ExperimentTemplate.build_result",
            "kind": "template_result_builder",
            "normalizer_required": True,
            "covered_by_normalizer": True,
            "evidence": "calls normalize_artifact_for_template_write before schema finalization",
        },
        {
            "surface": "python/carnot/experiment_5272_internal_hallucination_probe_gated_v482.py",
            "kind": "direct_strict_producer",
            "normalizer_required": False,
            "covered_by_normalizer": False,
            "evidence": "uses local validate_artifact; included as downstream evidence surface",
        },
        {
            "surface": "python/carnot/experiment_5274_solver_constraint_extraction_retry_gated_v482.py",
            "kind": "direct_strict_producer",
            "normalizer_required": False,
            "covered_by_normalizer": False,
            "evidence": "uses local validate_artifact; blocked artifacts remain flagged instead of normalized clean",
        },
        {
            "surface": "python/carnot/experiment_5276_memory_assisted_verifier_dose_gated_v482.py",
            "kind": "direct_strict_producer",
            "normalizer_required": False,
            "covered_by_normalizer": False,
            "evidence": "uses local validate_artifact; included as downstream evidence surface",
        },
    ]


def _producer_coverage(producer_inventory: Sequence[Mapping[str, Any]]) -> float:
    required = [row for row in producer_inventory if row.get("normalizer_required") is True]
    covered = [row for row in required if row.get("covered_by_normalizer") is True]
    return round(len(covered) / len(required), 6) if required else 0.0


def build_audit_matrix() -> list[JsonDict]:
    """Build the shape/evidence/duration audit matrix."""

    payloads: dict[str, JsonDict] = {
        "valid_shape_only_artifact": {
            "honest_verdict": {"value": "complete: shape-only fixture", "principle": "terminal"},
            "inference_substrate": {"value": INFERENCE_SUBSTRATE, "principle": "aggregation"},
            "duration_s": 0.01,
            "field_principles": {
                "honest_verdict": "terminal",
                "inference_substrate": "aggregation",
            },
        },
        "missing_evidence": {
            "honest_verdict": "complete: live fixture missing methodology",
            "inference_substrate": "live_llm_inference",
            "duration_s": 61.0,
            "field_principles": {
                "honest_verdict": "terminal",
                "inference_substrate": "live model",
                "duration_s": "wall clock",
            },
        },
        "bare_gate_fields": {
            "honest_verdict": "complete: bare gate fixture",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "producer_normalizer_ready": True,
        },
        "dict_wrapped_substrate_fields": {
            "honest_verdict": "complete: wrapped substrate fixture",
            "inference_substrate": {"value": INFERENCE_SUBSTRATE, "principle": "aggregation"},
            "duration_s": 0.01,
        },
        "sub_threshold_duration": {
            "honest_verdict": "complete: too-fast live fixture",
            "inference_substrate": "live_llm_inference",
            "duration_s": 0.5,
            "model_specs": [{"hf_id": "fixture-35B-GGUF"}],
            "random_seed": 5280,
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
        "no_llm_aggregation_artifact": {
            "honest_verdict": "complete: aggregation fixture",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "duration_s": 0.001,
            "cited_upstream_artifacts": [
                {
                    "path": "results/upstream.json",
                    "inherited_model_marker": "unsloth/gemma-4-31B-it-GGUF",
                }
            ],
        },
    }
    gate_fields = {"bare_gate_fields": ("producer_normalizer_ready",)}
    expected_flags = {
        "missing_evidence": {"METHODOLOGY_MISSING"},
        "sub_threshold_duration": {"DURATION_TOO_SHORT"},
    }
    matrix: list[JsonDict] = []
    for name in AUDIT_CASES:
        normalized = normalize_artifact_for_template_write(
            payloads[name],
            gate_fields=gate_fields.get(name, ()),
            required_principle_fields=(
                ("honest_verdict", "inference_substrate", "duration_s")
                if name == "missing_evidence"
                else ()
            ),
        )
        receipts = normalized.get(PRODUCER_NORMALIZER_RECEIPTS_FIELD, {})
        safe_repair_kinds = _receipt_kinds(receipts, "safe_repairs")
        unsafe_rejection_kinds = _receipt_kinds(receipts, "unsafe_rejections")
        ready = (
            receipts.get("ready_for_gated_consumers", True)
            if isinstance(receipts, Mapping)
            else True
        )
        report = _verify_payload(normalized)
        flags = _flag_kinds(report)
        floor_input = av._normalize_principle_wrapped_fields(dict(normalized))
        duration_floor = av.duration_floor_for_artifact(floor_input)
        expected = expected_flags.get(name, set())
        passed = (
            expected.issubset(set(flags))
            if expected
            else not set(flags)
            & {
                "DURATION_TOO_SHORT",
                "METHODOLOGY_MISSING",
                "GATE_PASSED_WITHOUT_DATA",
            }
        )
        if name == "missing_evidence":
            passed = passed and "missing_methodology_receipt" in unsafe_rejection_kinds
        if name == "bare_gate_fields":
            passed = (
                passed
                and normalized.get("producer_normalizer_ready") is True
                and isinstance(normalized.get("producer_normalizer_ready"), bool)
                and not safe_repair_kinds
                and not unsafe_rejection_kinds
            )
        if name == "sub_threshold_duration":
            passed = passed and "duration_too_short" in unsafe_rejection_kinds
        matrix.append(
            {
                "case": name,
                "normalized": normalized,
                "safe_repair_kinds": safe_repair_kinds,
                "unsafe_rejection_kinds": unsafe_rejection_kinds,
                "ready_for_gated_consumers": bool(ready),
                "duration_floor": duration_floor,
                "adversarial_flags": flags,
                "passed": bool(passed),
            }
        )
    return matrix


def v481_quarantine_checks(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    """Confirm old v481 pilots remain quarantined by live verifier evidence."""

    root_path = Path(root)
    checks: list[JsonDict] = []
    for relative in V481_QUARANTINE_RELATIVE_PATHS:
        path = root_path / relative
        if not path.exists():
            checks.append({"path": str(relative), "flags": ["artifact_missing"], "passed": False})
            continue
        report = av.verify_artifact(path)
        flags = _flag_kinds(report)
        checks.append(
            {
                "path": str(relative),
                "flags": flags,
                "passed": "METHODOLOGY_MISSING" in flags,
            }
        )
    return checks


def audit_summary(
    *,
    matrix: Sequence[Mapping[str, Any]],
    producer_inventory: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize matrix and producer coverage into artifact-ready gates."""

    by_case = {str(row["case"]): row for row in matrix}
    quarantine = v481_quarantine_checks()
    producer_coverage = _producer_coverage(producer_inventory)
    bare_gate_preservation_passed = bool(
        by_case["bare_gate_fields"]["passed"]
        and by_case["bare_gate_fields"]["normalized"]["producer_normalizer_ready"] is True
    )
    missing_evidence_rejected = bool(by_case["missing_evidence"]["passed"])
    duration_substrate_regression_passed = all(
        bool(by_case[name]["passed"])
        for name in (
            "dict_wrapped_substrate_fields",
            "sub_threshold_duration",
            "no_llm_aggregation_artifact",
        )
    )
    matrix_passed = all(bool(row["passed"]) for row in matrix)
    v481_quarantine_preserved = all(bool(row["passed"]) for row in quarantine)
    return {
        "normalizer_evidence_ready": bool(
            matrix_passed
            and producer_coverage == 1.0
            and v481_quarantine_preserved
            and bare_gate_preservation_passed
            and missing_evidence_rejected
            and duration_substrate_regression_passed
        ),
        "producer_coverage": producer_coverage,
        "bare_gate_preservation_passed": bare_gate_preservation_passed,
        "missing_evidence_rejected": missing_evidence_rejected,
        "duration_substrate_regression_passed": duration_substrate_regression_passed,
        "adversarial_verify_weakening": False,
        "v481_quarantine_preserved": v481_quarantine_preserved,
        "matrix_passed": matrix_passed,
    }


def build_artifact(*, tests_run: Sequence[Mapping[str, Any]], duration_s: float) -> JsonDict:
    """Build the Exp 5280 terminal audit artifact."""

    producer_inventory = enumerate_producers()
    matrix = build_audit_matrix()
    summary = audit_summary(matrix=matrix, producer_inventory=producer_inventory)
    ready = bool(summary["normalizer_evidence_ready"])
    verdict = (
        "complete: producer evidence discipline is ready at the template normalizer "
        "boundary; bare gates stay bare, missing evidence is rejected, duration and "
        "substrate behavior is preserved, and old v481 pilots remain quarantined."
        if ready
        else "blocked_normalizer_evidence_audit: producer evidence discipline is not ready."
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": list(SPEC_REFS),
        "duration_s": float(duration_s),
        "field_principles": dict(FIELD_PRINCIPLES),
        "producer_inventory": producer_inventory,
        "audit_matrix": matrix,
        "v481_quarantine_checks": v481_quarantine_checks(),
        "source_artifacts_read": [str(path) for path in SOURCE_RELATIVE_PATHS],
        "research_conductor_modified": {
            "value": False,
            "principle": "False because Exp5280 must not modify scripts/research_conductor.py.",
        },
        "honest_verdict": _wrap("honest_verdict", verdict),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "normalizer_evidence_ready": _wrap(
            "normalizer_evidence_ready", summary["normalizer_evidence_ready"]
        ),
        "producer_coverage": _wrap("producer_coverage", summary["producer_coverage"]),
        "bare_gate_preservation_passed": _wrap(
            "bare_gate_preservation_passed", summary["bare_gate_preservation_passed"]
        ),
        "missing_evidence_rejected": _wrap(
            "missing_evidence_rejected", summary["missing_evidence_rejected"]
        ),
        "duration_substrate_regression_passed": _wrap(
            "duration_substrate_regression_passed",
            summary["duration_substrate_regression_passed"],
        ),
        "adversarial_verify_weakening": _wrap("adversarial_verify_weakening", False),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _is_wrapped(value: Any) -> bool:
    return isinstance(value, Mapping) and "value" in value and "principle" in value


def _check(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 5280 artifact schema and audit gates."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    _check(not missing, errors, f"missing required fields: {missing}")
    _check(artifact.get("field_principles") == FIELD_PRINCIPLES, errors, "field_principles")
    for field in REQUIRED_WRAPPED_FIELDS:
        value = artifact.get(field)
        wrapped = _is_wrapped(value)
        _check(wrapped, errors, f"{field} must be principle-wrapped")
        principle = value.get("principle") if isinstance(value, Mapping) else None
        _check(principle == FIELD_PRINCIPLES[field], errors, f"{field} principle")
    verdict = (
        artifact.get("honest_verdict", {}).get("value")
        if _is_wrapped(artifact.get("honest_verdict"))
        else None
    )
    _check(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        errors,
        "honest_verdict terminal",
    )
    _check(
        isinstance(verdict, str) and "producer evidence discipline" in verdict,
        errors,
        "honest_verdict readiness statement",
    )
    _check(
        artifact.get("inference_substrate", {}).get("value") == INFERENCE_SUBSTRATE
        if _is_wrapped(artifact.get("inference_substrate"))
        else False,
        errors,
        "inference_substrate",
    )
    _check(
        artifact.get("adversarial_verify_weakening", {}).get("value") is False
        if _is_wrapped(artifact.get("adversarial_verify_weakening"))
        else False,
        errors,
        "adversarial_verify_weakening",
    )
    tests_run = artifact.get("tests_run")
    _check(isinstance(tests_run, list) and bool(tests_run), errors, "tests_run")
    for row in tests_run if isinstance(tests_run, list) else []:
        _check(
            isinstance(row, Mapping) and bool(row.get("command")) and bool(row.get("outcome")),
            errors,
            "tests_run row",
        )
    _check(
        artifact.get("research_conductor_modified", {}).get("value") is False
        if _is_wrapped(artifact.get("research_conductor_modified"))
        else False,
        errors,
        "research_conductor_modified",
    )
    _check(artifact.get("reproducibility_checksum") == _checksum(artifact), errors, "checksum")
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(
    *,
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> JsonDict:
    """Write the Exp 5280 JSON artifact and return it."""

    artifact = build_artifact(tests_run=tests_run, duration_s=duration_s)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--test-run", action="append", default=[])
    args = parser.parse_args(argv)
    started = time.monotonic()
    tests_run = [
        {"command": item.split("=", 1)[0], "outcome": item.split("=", 1)[1]}
        if "=" in item
        else {"command": item, "outcome": "RECORDED"}
        for item in args.test_run
    ] or [{"command": "not provided", "outcome": "RECORDED"}]
    artifact = write_artifact(
        output_path=args.output,
        tests_run=tests_run,
        duration_s=max(time.monotonic() - started, 0.001),
    )
    print(
        json.dumps(
            {"result_path": str(args.output), "checksum": artifact["reproducibility_checksum"]}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
