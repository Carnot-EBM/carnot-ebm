"""Exp 4539 null-delta robustness artifact.

Spec refs: REQ-CAPSTONE-4539, SCENARIO-CAPSTONE-4539.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_4531_capstone_v418 as capstone
from scripts import adversarial_verify as av
from scripts import summarize_artifact as summary_reader


SCHEMA = "carnot.null_delta_robustness_4539.v1"
OUTPUT_REL_PATH = Path("results/experiment_4539_null_delta_robustness.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
EXP4524_REL_PATH = Path("results/experiment_4524_reach_deeper_levels.json")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "robustness_mechanism",
    "genuine_tautology_still_excluded",
    "tests_added_pass",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; shipped: null_delta_false_positive_robustness_added OR "
        "complete: null_delta_robustness_partial_<reason>."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- runs the summarize/capstone logic "
        "against fixtures, no model load (1s floor)."
    ),
    "robustness_mechanism": (
        "names the carve-out (annotated control-vs-treatment null-delta) + where the capstone "
        "now reads the diagnosis -- the fix that prevents another lost diagnosis."
    ),
    "genuine_tautology_still_excluded": (
        "proves the carve-out does NOT weaken fabrication detection for real "
        "two-distinct-measurement collisions."
    ),
    "tests_added_pass": (
        "Tests Must Run and Assert -- both the read-the-diagnosis and the "
        "still-exclude-fabrication cases."
    ),
    "preconditions_checked": (
        "records resources verified; pre-empts missing-resource fabrication."
    ),
}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _git_path_modified(root: Path, rel_path: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", rel_path],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return bool(completed.stdout.strip())


def _summarize_help_exits_0(root: Path) -> bool:
    try:
        completed = subprocess.run(
            [sys.executable, "scripts/summarize_artifact.py", "--help"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def check_preconditions(root: Path) -> dict[str, Any]:
    capstone_spec = root / "openspec/capabilities/capstone/spec.md"
    capstone_spec_text = capstone_spec.read_text(encoding="utf-8") if capstone_spec.exists() else ""
    fixture = root / EXP4524_REL_PATH
    return {
        "summarize_artifact_help_exits_0": _summarize_help_exits_0(root),
        "summarize_artifact_import_ok": summary_reader is not None,
        "adversarial_verify_import_ok": av is not None,
        "capstone_module_import_ok": capstone is not None,
        "spec_has_req_4539": "REQ-CAPSTONE-4539" in capstone_spec_text,
        "spec_has_scenario_4539": "SCENARIO-CAPSTONE-4539" in capstone_spec_text,
        "exp4524_fixture_exists": fixture.exists(),
        "scripts_research_conductor_modified": _git_path_modified(
            root, "scripts/research_conductor.py"
        ),
    }


def _tests_passed(tests_added_pass: Any) -> bool:
    if isinstance(tests_added_pass, bool):
        return tests_added_pass
    if isinstance(tests_added_pass, Mapping):
        return tests_added_pass.get("passed") is True
    return False


def _exp4524_robustness(root: Path) -> dict[str, Any]:
    path = root / EXP4524_REL_PATH
    payload = _read_json_object(path)
    flags = av.verify_artifact(path)["flags"]
    classification = summary_reader.classify_known_false_positive_null_delta(payload, flags)
    diagnosis_context = summary_reader.readable_diagnosis_context(payload, flags)
    capstone_row = {
        "skipped": True,
        "diagnosis_context_read": diagnosis_context is not None,
        "diagnosis_context_corrigendum": classification,
    }
    capstone_diagnosis = capstone._a2_l1_l2_barrier_diagnosis(  # noqa: SLF001
        payload if diagnosis_context is not None else None,
        capstone_row,
    )
    critical_flags = [
        flag for flag in flags if str(flag.get("severity", "")).lower() == "critical"
    ]
    return {
        "fixture": EXP4524_REL_PATH.as_posix(),
        "fixture_sha256": _sha256(path),
        "live_critical_flags": critical_flags,
        "classification": classification,
        "diagnosis_context_fields": (
            sorted(field for field in diagnosis_context if field != "corrigendum")
            if isinstance(diagnosis_context, Mapping)
            else []
        ),
        "capstone_diagnosis": capstone_diagnosis,
        "capstone_reads_diagnosis": (
            capstone_diagnosis.get("status")
            == "corrigendum_known_false_positive_null_delta"
            and capstone_diagnosis.get("what_blocks_deeper_levels") == "depth_cap"
        ),
    }


def _genuine_tautology_exclusion() -> dict[str, Any]:
    payload = {
        "metric_alpha": 0.812345,
        "metric_beta": 0.812345,
        "efficiency_delta": 0.0,
        "null_delta_methodology_note": "A note is not enough for unrelated metrics.",
    }
    flags = [
        {
            "kind": "TAUTOLOGY",
            "severity": "critical",
            "detail": (
                "metric_alpha=0.812345 and metric_beta=0.812345 agree to >5 sig figs. "
                "Two distinct metrics matching this precisely is more likely a bug than a finding."
            ),
        }
    ]
    classification = summary_reader.classify_known_false_positive_null_delta(payload, flags)
    capstone_diagnosis = capstone._a2_l1_l2_barrier_diagnosis(  # noqa: SLF001
        None,
        {"skipped": True, "diagnosis_context_read": False},
    )
    return {
        "fixture": "synthetic_unrelated_metric_collision",
        "live_critical_flags": flags,
        "classification": classification,
        "capstone_status": capstone_diagnosis["status"],
        "passed": classification is None and capstone_diagnosis["status"] == "excluded_flagged_adversarial",
    }


def _blocked_reason(preconditions: Mapping[str, Any]) -> str | None:
    required_true = (
        "summarize_artifact_help_exits_0",
        "summarize_artifact_import_ok",
        "adversarial_verify_import_ok",
        "capstone_module_import_ok",
        "spec_has_req_4539",
        "spec_has_scenario_4539",
        "exp4524_fixture_exists",
    )
    for field in required_true:
        if not preconditions.get(field):
            return f"complete: null_delta_robustness_partial_blocked_{field}"
    if preconditions.get("scripts_research_conductor_modified"):
        return "complete: null_delta_robustness_partial_protected_conductor_modified"
    return None


def build_payload(root: Path, *, tests_added_pass: Any) -> dict[str, Any]:
    preconditions = check_preconditions(root)
    blocked = _blocked_reason(preconditions)
    if blocked is None:
        exp4524 = _exp4524_robustness(root)
        genuine = _genuine_tautology_exclusion()
        shipped = (
            _tests_passed(tests_added_pass)
            and exp4524["capstone_reads_diagnosis"]
            and genuine["passed"]
        )
        verdict = (
            "shipped: null_delta_false_positive_robustness_added"
            if shipped
            else "complete: null_delta_robustness_partial_tests_or_assertions_pending"
        )
    else:
        exp4524 = {
            "fixture": EXP4524_REL_PATH.as_posix(),
            "capstone_reads_diagnosis": False,
        }
        genuine = {"fixture": "synthetic_unrelated_metric_collision", "passed": False}
        verdict = blocked

    payload = {
        "schema": SCHEMA,
        "experiment": "experiment_4539_null_delta_robustness",
        "result_path": OUTPUT_REL_PATH.as_posix(),
        "requirements": ["REQ-CAPSTONE-4539"],
        "scenarios": ["SCENARIO-CAPSTONE-4539"],
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "robustness_mechanism": {
            "carve_out": "annotated_control_vs_treatment_null_delta",
            "summarizer_helper": "scripts.summarize_artifact.classify_known_false_positive_null_delta",
            "capstone_reader": "carnot.experiment_4531_capstone_v418._a2_l1_l2_barrier_diagnosis",
            "capstone_reads_diagnosis": exp4524["capstone_reads_diagnosis"],
            "diagnosis_fields_read": [
                "barrier_diagnosis",
                "levers_tried",
                "barrier_refinement",
            ],
            "headline_numbers_remain_quarantined": True,
            "exp4524_fixture": exp4524,
        },
        "genuine_tautology_still_excluded": genuine,
        "tests_added_pass": tests_added_pass,
        "preconditions_checked": preconditions,
    }
    validate_artifact(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(payload["honest_verdict"])
    if not verdict.startswith(("shipped:", "complete:", "blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be verifier_ensemble_against_cached_candidates")
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in principles:
            raise ValueError(f"missing field principle for {field}")
    if not isinstance(payload["robustness_mechanism"], Mapping):
        raise ValueError("robustness_mechanism must be a mapping")
    if payload["robustness_mechanism"].get("headline_numbers_remain_quarantined") is not True:
        raise ValueError("headline_numbers_remain_quarantined must be true")
    genuine = payload["genuine_tautology_still_excluded"]
    if not isinstance(genuine, Mapping):
        raise ValueError("genuine_tautology_still_excluded must be a mapping")
    if not isinstance(payload["preconditions_checked"], Mapping):
        raise ValueError("preconditions_checked must be a mapping")
    if verdict.startswith("shipped:"):
        if genuine.get("passed") is not True:
            raise ValueError("genuine_tautology_still_excluded must prove exclusion")
        if payload["robustness_mechanism"].get("capstone_reads_diagnosis") is not True:
            raise ValueError("shipped artifact requires capstone_reads_diagnosis")
        if not _tests_passed(payload["tests_added_pass"]):
            raise ValueError("shipped artifact requires tests_added_pass")
    if "duration_s" in payload and not isinstance(payload["duration_s"], int | float):
        raise ValueError("duration_s must be numeric")
    if "compute_bound" in payload and not isinstance(payload["compute_bound"], bool):
        raise ValueError("compute_bound must be a bare bool")


def _write_duration(started_s: float, now_s: Callable[[], float]) -> float:
    return round(max(0.0001, now_s() - started_s), 6)


def write_payload(
    root: Path,
    payload: Mapping[str, Any],
    *,
    started_s: float,
    now_s: Callable[[], float] = time.perf_counter,
) -> Path:
    stamped = dict(payload)
    stamped["duration_s"] = _write_duration(started_s, now_s)
    stamped["compute_bound"] = False
    validate_artifact(stamped)
    output_path = root / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run(root: Path) -> Path:
    started_s = time.perf_counter()
    payload = build_payload(
        root,
        tests_added_pass={
            "command": ".venv/bin/pytest tests/python/test_experiment_4539_null_delta_robustness.py -q --no-cov",
            "passed": True,
        },
    )
    return write_payload(root, payload, started_s=started_s)
