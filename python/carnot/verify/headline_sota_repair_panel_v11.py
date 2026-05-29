"""Run the Exp 3302 headline SOTA repair panel v11 artifact.

Spec refs: REQ-VERIFY-3302, SCENARIO-VERIFY-3302.

This module consumes the fixed Exp 3301 exact manifest as the denominator. It
generates one repair per case with an available mandated local GGUF model, then
counts success only when two independent gates agree: the calibrated clean
verifier says ACCEPT and the deterministic exact checker passes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
Probe = Callable[[], JsonDict]
CandidateRunner = Callable[[list[JsonDict], JsonDict, int], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.headline_sota_repair_panel.v11"
EXPERIMENT_ID = "exp3302"
TASK_ID = "exp3302-headline-sota-repair-panel-v11"
ARTIFACT = "experiment_3302_headline_sota_repair_panel_v11"
MILESTONE = "2026.05.305"
RUN_DATE = "20260529"
RANDOM_SEED = 3302

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3302_headline_sota_repair_panel_v11.json")
EXP3300_REL_PATH = Path("results/experiment_3300_full_garak_dataflip_gate_rerun_v3.json")
EXP3301_REL_PATH = Path("results/experiment_3301_exact_repair_panel_manifest_v11.json")
EXP3287_REL_PATH = Path("results/experiment_3287_abstention_calibrated_clean_verifier_v15.json")
PANEL_CASES_REL_PATH = Path("data/research/exact_repair_panel_v11.jsonl")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
DECISION_GRAMMAR = 'root ::= "ACCEPT" | "REJECT" | "ABSTAIN"\n'
MIN_PANEL_CASES = 30

REQUIRED_FIELDS = {
    "headline_repair_panel_ready",
    "repair_panel_ran",
    "headline_claim_allowed",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "panel_case_count",
    "verified_success_count",
    "repair_success_rate",
    "repair_success_ci95",
    "false_accept_count",
    "false_accept_rate_ci95",
    "abstention_count",
    "per_family_metrics",
    "candidate_results",
    "gpu_mem_used_mib",
    "tokens_generated",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3302_headline_sota_repair_panel_v11.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/headline_sota_repair_panel_v11.py -m pytest -o addopts='' tests/python/test_experiment_3302_headline_sota_repair_panel_v11.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/headline_sota_repair_panel_v11.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)


@dataclass(frozen=True)
class JsonLoad:
    """A JSON source plus read diagnostics that can be embedded in artifacts."""

    payload: JsonDict
    present: bool
    readable: bool
    error: str | None
    path: Path
    sha256: str | None


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    nvidia_probe: Probe | None = None,
    python_cuda_probe: Probe | None = None,
    candidate_runner: CandidateRunner | None = None,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3302: build the headline repair panel or a terminal skip artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)

    exp3300_load = read_json_object(root_path / EXP3300_REL_PATH)
    exp3301_load = read_json_object(root_path / EXP3301_REL_PATH)
    exp3287_load = read_json_object(root_path / EXP3287_REL_PATH)
    manifest_path = manifest_cases_path(root_path, exp3301_load.payload)
    manifest_rows = read_jsonl_objects(manifest_path)
    manifest_check = validate_manifest_contract(exp3301_load, manifest_rows)
    clean_check = clean_verifier_contract(exp3287_load)

    preconditions: list[JsonDict] = [
        exp3300_precondition(exp3300_load),
        exp3301_precondition(exp3301_load, manifest_check),
        exp3287_precondition(exp3287_load, clean_check),
    ]
    nvidia = normalize_precondition((nvidia_probe or default_nvidia_smi_probe)())
    python_cuda = normalize_precondition((python_cuda_probe or selected_python_cuda_probe)())
    preconditions.extend([nvidia, python_cuda])

    inventory = resolve_mandated_model_inventory(root_path)
    preconditions.append(model_cache_precondition(inventory))
    blocked_reasons = precondition_blockers(preconditions)
    model = select_model_for_panel(inventory)

    runner_payload: JsonDict = {}
    evaluated_rows: list[JsonDict] = []
    runner_error = ""
    operationally_ready = not blocked_reasons and model is not None and manifest_check["valid"] is True
    if operationally_ready:
        try:
            runner = candidate_runner or run_llama_repair_panel
            runner_payload = mapping(runner(manifest_rows, model, int(random_seed)))
            evaluated_rows = evaluate_runner_rows(
                manifest_rows,
                runner_payload.get("rows"),
                model,
            )
        except Exception as exc:  # pragma: no cover - defensive live-run boundary.
            runner_error = f"{type(exc).__name__}: {exc}"
            blocked_reasons.append("repair_candidate_runner_failed")

    metrics = panel_metrics(evaluated_rows)
    gpu_mem_used_mib = safe_int(runner_payload.get("gpu_mem_used_mib"))
    tokens_generated = count_generated_tokens(evaluated_rows, runner_payload)
    repair_panel_ran = bool(evaluated_rows) and not runner_error
    panel_case_count = len(evaluated_rows) if repair_panel_ran else 0
    provenance_clean = source_clean(
        exp3300_load.payload,
        exp3301_load.payload,
        exp3287_load.payload,
    )
    headline_ready = (
        repair_panel_ran
        and panel_case_count >= MIN_PANEL_CASES
        and panel_case_count == len(manifest_rows)
        and gpu_mem_used_mib > 0
        and tokens_generated > 0
    )
    claim_allowed = (
        headline_ready
        and metrics["false_accept_count"] == 0
        and provenance_clean
        and all(row.get("passed") is True for row in preconditions)
        and bool(models_used(model, runner_payload, panel_case_count))
    )
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3302", "SCENARIO-VERIFY-3302"],
        "headline_repair_panel_ready": headline_ready,
        "repair_panel_ran": repair_panel_ran,
        "headline_claim_allowed": claim_allowed,
        "model_specs": model_specs_from_inventory(inventory, clean_check),
        "models_used": models_used(model, runner_payload, panel_case_count) if repair_panel_ran else [],
        "missing_model_specs": mapping_list(inventory.get("missing_model_specs")),
        "preconditions_checked": preconditions,
        "panel_case_count": panel_case_count,
        "verified_success_count": metrics["verified_success_count"],
        "repair_success_rate": metrics["repair_success_rate"],
        "repair_success_ci95": wilson_ci95(metrics["verified_success_count"], panel_case_count),
        "false_accept_count": metrics["false_accept_count"],
        "false_accept_rate": metrics["false_accept_rate"],
        "false_accept_rate_ci95": wilson_ci95(metrics["false_accept_count"], panel_case_count),
        "abstention_count": metrics["abstention_count"],
        "per_family_metrics": per_family_metrics(evaluated_rows),
        "candidate_results": evaluated_rows,
        "gpu_mem_used_mib": gpu_mem_used_mib if repair_panel_ran else 0,
        "tokens_generated": tokens_generated if repair_panel_ran else 0,
        "inference_substrate": (
            "live_local_sota_gguf_repair_plus_calibrated_clean_verifier"
            if repair_panel_ran
            else "gated_skip_or_precondition_block"
        ),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration(started, finished),
        "honest_verdict": "",
        "blocked_reasons": sorted(set(blocked_reasons)),
        "runner_error": runner_error,
        "provenance_clean": provenance_clean,
        "manifest_case_hashes": [str(row.get("case_hash") or "") for row in manifest_rows],
        "manifest_case_hashes_match": manifest_check["case_hashes_match"],
        "manifest_cases_path": relative_or_abs(root_path, manifest_path),
        "source_artifacts": [
            source_artifact_row(exp3300_load, "exp3300_garak_gate"),
            source_artifact_row(exp3301_load, "exp3301_fixed_manifest"),
            source_artifact_row(exp3287_load, "exp3287_clean_verifier"),
            file_source_artifact_row(manifest_path, "fixed_panel_cases_jsonl"),
        ],
        "clean_verifier_policy": clean_check["policy"],
        "case_list_frozen_before_generation": True,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    nvidia_probe: Probe | None = None,
    python_cuda_probe: Probe | None = None,
    candidate_runner: CandidateRunner | None = None,
    random_seed: int = RANDOM_SEED,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3302 terminal JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(
        root_path,
        nvidia_probe=nvidia_probe,
        python_cuda_probe=python_cuda_probe,
        candidate_runner=candidate_runner,
        random_seed=random_seed,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_object(path: Path) -> JsonLoad:
    """Read a JSON object while preserving missing or malformed source evidence."""

    if not path.is_file():
        return JsonLoad({}, False, False, "missing", path, None)
    digest = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return JsonLoad({}, True, False, str(exc), path, digest)
    if not isinstance(payload, Mapping):
        return JsonLoad({}, True, False, "json root is not an object", path, digest)
    return JsonLoad(dict(payload), True, True, None, path, digest)


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Read JSONL rows and drop malformed/non-object rows without guessing."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def manifest_cases_path(root: Path, exp3301: Mapping[str, Any]) -> Path:
    """Return the fixed panel JSONL path named by Exp 3301, defaulting to v11."""

    path = Path(str(exp3301.get("panel_cases_path") or PANEL_CASES_REL_PATH.as_posix()))
    return path if path.is_absolute() else root / path


def validate_manifest_contract(exp3301_load: JsonLoad, cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check that the fixed manifest can be used without changing its denominator."""

    payload = exp3301_load.payload
    expected_hashes = [str(value) for value in sequence(payload.get("case_hashes"))]
    observed_hashes = [str(case.get("case_hash") or "") for case in cases]
    case_hashes_match = bool(expected_hashes) and expected_hashes == observed_hashes
    known_failing_count = sum(
        not exact_check(case, str(case.get("failing_candidate") or "")) for case in cases
    )
    expected_pass_count = sum(
        exact_check(case, str(case.get("expected_answer") or "")) for case in cases
    )
    feedback_count = sum(bool(str(case.get("localized_repair_feedback") or "").strip()) for case in cases)
    valid = (
        exp3301_load.present
        and exp3301_load.readable
        and payload.get("repair_panel_manifest_ready") is True
        and len(cases) >= MIN_PANEL_CASES
        and safe_int(payload.get("panel_case_count")) == len(cases)
        and case_hashes_match
        and known_failing_count == len(cases)
        and expected_pass_count == len(cases)
        and feedback_count == len(cases)
        and all(case.get("llm_judge_required") is not True for case in cases)
    )
    return {
        "valid": valid,
        "case_hashes_match": case_hashes_match,
        "known_failing_candidate_count": known_failing_count,
        "expected_pass_count": expected_pass_count,
        "localized_feedback_count": feedback_count,
    }


def exp3300_precondition(exp3300_load: JsonLoad) -> JsonDict:
    """Expose the upstream Garak gate that permits this repair panel to run."""

    passed = (
        exp3300_load.present
        and exp3300_load.readable
        and exp3300_load.payload.get("garak_gate_passed") is True
    )
    return {
        "name": "exp3300_garak_gate",
        "passed": passed,
        "path": EXP3300_REL_PATH.as_posix(),
        "present": exp3300_load.present,
        "readable": exp3300_load.readable,
        "error": exp3300_load.error,
        "garak_gate_passed": exp3300_load.payload.get("garak_gate_passed") is True,
        "sha256": exp3300_load.sha256,
    }


def exp3301_precondition(exp3301_load: JsonLoad, manifest_check: Mapping[str, Any]) -> JsonDict:
    """Expose the fixed exact manifest condition and its frozen case-hash check."""

    return {
        "name": "exp3301_fixed_exact_manifest",
        "passed": manifest_check.get("valid") is True,
        "path": EXP3301_REL_PATH.as_posix(),
        "present": exp3301_load.present,
        "readable": exp3301_load.readable,
        "error": exp3301_load.error,
        "repair_panel_manifest_ready": exp3301_load.payload.get("repair_panel_manifest_ready")
        is True,
        "panel_case_count": safe_int(exp3301_load.payload.get("panel_case_count")),
        "case_hashes_match": manifest_check.get("case_hashes_match") is True,
        "sha256": exp3301_load.sha256,
    }


def clean_verifier_contract(exp3287_load: JsonLoad) -> JsonDict:
    """Check the Exp 3287 calibrated clean-verifier contract used for accept gates."""

    payload = exp3287_load.payload
    policy = mapping(payload.get("calibrated_abstention_policy"))
    ready = (
        exp3287_load.present
        and exp3287_load.readable
        and payload.get("abstention_calibrated_clean_verifier_v15_ready") is True
        and payload.get("clean_verifier_rerun_ready") is True
        and payload.get("repair_gate_input_clean_enough") is True
        and safe_int(payload.get("false_accept_count")) == 0
        and str(policy.get("grammar") or "") == "ACCEPT|REJECT|ABSTAIN"
    )
    return {"ready": ready, "policy": policy}


def exp3287_precondition(exp3287_load: JsonLoad, clean_check: Mapping[str, Any]) -> JsonDict:
    """Expose clean verifier readiness without making the verifier self-authoritative."""

    return {
        "name": "exp3287_abstention_calibrated_clean_verifier",
        "passed": clean_check.get("ready") is True,
        "path": EXP3287_REL_PATH.as_posix(),
        "present": exp3287_load.present,
        "readable": exp3287_load.readable,
        "error": exp3287_load.error,
        "clean_verifier_rerun_ready": exp3287_load.payload.get("clean_verifier_rerun_ready")
        is True,
        "repair_gate_input_clean_enough": exp3287_load.payload.get("repair_gate_input_clean_enough")
        is True,
        "false_accept_count": safe_int(exp3287_load.payload.get("false_accept_count")),
        "sha256": exp3287_load.sha256,
    }


def resolve_mandated_model_inventory(root: Path) -> JsonDict:
    """Resolve all mandated GGUF paths, trying cached_sota_pair before single-model fallback."""

    pair = cached_sota_pair(gpu_indices=(0, 1))
    pair_by_id = {str(row.get("hf_id") or row.get("model_id") or ""): row for row in pair or []}
    specs_by_id = {spec["hf_id"]: spec for spec in SOTA_GGUF_MODELS}
    available: list[JsonDict] = []
    missing: list[JsonDict] = []
    mandated_models: JsonDict = {}
    for model_id in MANDATED_MODEL_IDS:
        spec = mapping(specs_by_id.get(model_id))
        pair_entry = mapping(pair_by_id.get(model_id))
        resolved = str(pair_entry.get("model_path") or resolve_cached_gguf(model_id) or "")
        path = resolve_path(root, resolved) if resolved else None
        cached = bool(path is not None and path.is_file() and path.stat().st_size > 0)
        record = {
            "model_id": model_id,
            "hf_id": model_id,
            "name": str(pair_entry.get("name") or spec.get("name") or model_id),
            "role": str(spec.get("role") or ""),
            "expected_quantization": str(spec.get("quantization") or "Q4_K_M"),
            "cached": cached,
            "model_path": str(path) if cached and path is not None else None,
            "size_bytes": int(path.stat().st_size) if cached and path is not None else 0,
        }
        mandated_models[model_id] = record
        if cached:
            available.append(
                record
                | {
                    "gpu": int(pair_entry.get("gpu", len(available) % 2)),
                    "source": "cached_sota_pair" if pair_entry else "resolve_cached_gguf",
                    "legacy_small_model": False,
                }
            )
        else:
            missing.append(
                {
                    "model_id": model_id,
                    "hf_id": model_id,
                    "name": record["name"],
                    "role": record["role"],
                    "expected_quantization": record["expected_quantization"],
                    "cached": False,
                    "model_path": None,
                    "reason": "not_cached",
                }
            )
    return {
        "cached_sota_pair_attempted": True,
        "cached_sota_pair_available": pair is not None,
        "cached_sota_pair_specs": [dict(row) for row in pair or []],
        "available_models": available,
        "missing_model_specs": missing,
        "mandated_models": mandated_models,
    }


def model_cache_precondition(inventory: Mapping[str, Any]) -> JsonDict:
    """Return the mandated GGUF cache check used before any model load."""

    return {
        "name": "mandated_sota_gguf_cache",
        "passed": bool(mapping_list(inventory.get("available_models"))),
        "cached_sota_pair_attempted": inventory.get("cached_sota_pair_attempted") is True,
        "cached_sota_pair_available": inventory.get("cached_sota_pair_available") is True,
        "available_model_ids": [
            row["model_id"] for row in mapping_list(inventory.get("available_models"))
        ],
        "missing_model_ids": [
            row["model_id"] for row in mapping_list(inventory.get("missing_model_specs"))
        ],
    }


def precondition_blockers(preconditions: Sequence[Mapping[str, Any]]) -> list[str]:
    """Translate failed operational preconditions into stable blocker strings."""

    names = [str(row.get("name") or "") for row in preconditions if row.get("passed") is not True]
    mapping_by_name = {
        "exp3300_garak_gate": "exp3300_garak_gate_not_passed",
        "exp3301_fixed_exact_manifest": "exp3301_fixed_manifest_unavailable",
        "exp3287_abstention_calibrated_clean_verifier": "exp3287_clean_verifier_unavailable",
        "nvidia_smi": "nvidia_smi_unavailable",
        "selected_python_cuda": "selected_python_cuda_unavailable",
        "mandated_sota_gguf_cache": "mandated_sota_gguf_unavailable",
    }
    return [mapping_by_name.get(name, f"{name}_failed") for name in names]


def select_model_for_panel(inventory: Mapping[str, Any]) -> JsonDict | None:
    """Pick the first available mandated GGUF in the user-specified order."""

    available = mapping_list(inventory.get("available_models"))
    return available[0] if available else None


def model_specs_from_inventory(
    inventory: Mapping[str, Any],
    clean_check: Mapping[str, Any],
) -> JsonDict:
    """Return the artifact model-spec block with mandated models always named."""

    return {
        "runtime": "llama_cpp_local_gguf_only",
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "cached_sota_pair_attempted": inventory.get("cached_sota_pair_attempted") is True,
        "cached_sota_pair_available": inventory.get("cached_sota_pair_available") is True,
        "cached_sota_pair_specs": mapping_list(inventory.get("cached_sota_pair_specs")),
        "available_model_count": len(mapping_list(inventory.get("available_models"))),
        "missing_model_count": len(mapping_list(inventory.get("missing_model_specs"))),
        "mandated_models": mapping(inventory.get("mandated_models")),
        "generation_runtime": "llama_cpp_local_generation",
        "verification_runtime": "exp3287_calibrated_accept_reject_abstain_contract",
        "calibrated_clean_verifier_policy": mapping(clean_check.get("policy")),
    }


def models_used(
    model: Mapping[str, Any] | None,
    runner_payload: Mapping[str, Any],
    n_cases: int,
) -> list[JsonDict]:
    """Describe the real model evidence used for the generated repair panel."""

    if not model:
        return []
    return [
        {
            "model_id": str(model.get("model_id") or model.get("hf_id") or ""),
            "hf_id": str(model.get("hf_id") or model.get("model_id") or ""),
            "name": str(model.get("name") or ""),
            "model_path": str(model.get("model_path") or ""),
            "role": str(model.get("role") or ""),
            "source": str(model.get("source") or ""),
            "legacy_small_model": model.get("legacy_small_model") is True,
            "repair_panel_case_count": int(n_cases),
            "gpu_mem_used_mib": safe_int(runner_payload.get("gpu_mem_used_mib")),
        }
    ]


def evaluate_runner_rows(
    cases: Sequence[Mapping[str, Any]],
    raw_rows: Any,
    model: Mapping[str, Any],
) -> list[JsonDict]:
    """Attach exact checks and calibrated verifier decisions to model repair rows."""

    raw_by_id = {
        str(row.get("case_id") or ""): mapping(row)
        for row in (raw_rows if isinstance(raw_rows, Sequence) else [])
        if isinstance(row, Mapping)
    }
    evaluated: list[JsonDict] = []
    for case in cases:
        raw = raw_by_id.get(str(case.get("case_id") or ""), {})
        candidate = clean_candidate_answer(raw.get("candidate_answer") or raw.get("output_text") or "")
        verifier_text = str(raw.get("verifier_output_text") or raw.get("verifier_decision") or "")
        verifier_decision = normalize_verifier_decision(verifier_text)
        exact_passed = exact_check(case, candidate)
        missing_candidate = candidate == ""
        abstained = verifier_decision == "abstain" or missing_candidate
        false_accept = verifier_decision == "accept" and not exact_passed
        verified_success = verifier_decision == "accept" and exact_passed and not missing_candidate
        evaluated.append(
            {
                "case_id": str(case.get("case_id") or ""),
                "family": str(case.get("family") or ""),
                "model_id": str(model.get("model_id") or model.get("hf_id") or ""),
                "model_path": str(model.get("model_path") or ""),
                "case_hash": str(case.get("case_hash") or ""),
                "failing_candidate": str(case.get("failing_candidate") or ""),
                "candidate_answer": candidate,
                "expected_answer": str(case.get("expected_answer") or ""),
                "exact_checker_type": str(case.get("exact_checker_type") or ""),
                "exact_check_passed": exact_passed,
                "calibrated_clean_verifier_decision": verifier_decision,
                "calibrated_clean_verifier_output": verifier_text,
                "verified_success": verified_success,
                "false_accept": false_accept,
                "abstained": abstained,
                "failure_class": failure_class(
                    exact_passed=exact_passed,
                    verifier_decision=verifier_decision,
                    missing_candidate=missing_candidate,
                ),
                "localized_repair_feedback": str(case.get("localized_repair_feedback") or ""),
                "candidate_hash": stable_hash(
                    {
                        "case_id": case.get("case_id"),
                        "candidate_answer": candidate,
                        "verifier_decision": verifier_decision,
                    }
                ),
                "token_counts": mapping(raw.get("token_counts")),
            }
        )
    return evaluated


def failure_class(
    *,
    exact_passed: bool,
    verifier_decision: str,
    missing_candidate: bool,
) -> str:
    """Classify one non-successful repair outcome for audit feedback."""

    if missing_candidate:
        return "missing_candidate_output"
    if verifier_decision == "abstain":
        return "clean_verifier_abstained"
    if verifier_decision == "accept" and not exact_passed:
        return "exact_mismatch_false_accept"
    if verifier_decision == "reject" and exact_passed:
        return "clean_verifier_rejected_exact_success"
    if verifier_decision == "reject":
        return "exact_mismatch_rejected"
    return "unknown_verifier_decision"


def panel_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute headline repair utility, false accepts, and abstentions."""

    n_rows = len(rows)
    verified_success_count = sum(row.get("verified_success") is True for row in rows)
    false_accept_count = sum(row.get("false_accept") is True for row in rows)
    abstention_count = sum(row.get("abstained") is True for row in rows)
    return {
        "verified_success_count": verified_success_count,
        "false_accept_count": false_accept_count,
        "abstention_count": abstention_count,
        "repair_success_rate": rate(verified_success_count, n_rows),
        "false_accept_rate": rate(false_accept_count, n_rows),
    }


def per_family_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute the repair panel metrics by manifest family."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family") or "unknown")].append(row)
    return {
        family: {
            "case_count": len(family_rows),
            "verified_success_count": sum(row.get("verified_success") is True for row in family_rows),
            "false_accept_count": sum(row.get("false_accept") is True for row in family_rows),
            "abstention_count": sum(row.get("abstained") is True for row in family_rows),
            "repair_success_rate": rate(
                sum(row.get("verified_success") is True for row in family_rows),
                len(family_rows),
            ),
        }
        for family, family_rows in sorted(grouped.items())
    }


def count_generated_tokens(
    rows: Sequence[Mapping[str, Any]],
    runner_payload: Mapping[str, Any],
) -> int:
    """Count completion tokens, falling back to a runner-level total when supplied."""

    row_total = sum(
        safe_int(mapping(row.get("token_counts")).get("completion_tokens")) for row in rows
    )
    if row_total:
        return row_total
    return safe_int(runner_payload.get("tokens_generated"))


def wilson_ci95(successes: int, n: int) -> list[float]:
    """Return a rounded Wilson 95% confidence interval for a binomial rate."""

    if n <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return [round(max(0.0, center - half_width), 6), round(min(1.0, center + half_width), 6)]


def exact_check(case: Mapping[str, Any], candidate_answer: str) -> bool:
    """Run the deterministic exact authority for one repaired candidate."""

    expected = clean_candidate_answer(case.get("expected_answer") or "")
    candidate = clean_candidate_answer(candidate_answer)
    checker = str(case.get("exact_checker_type") or "")
    if checker == "exact_integer_string":
        parsed_expected = parse_int_string(expected)
        parsed_candidate = parse_int_string(candidate)
        return parsed_candidate is not None and parsed_candidate == parsed_expected
    if checker == "exact_bool_string":
        parsed_expected = normalize_bool_string(expected)
        return parsed_expected in {"true", "false"} and normalize_bool_string(candidate) == parsed_expected
    if checker == "exact_stdout_string":
        return candidate.strip() == expected.strip()
    return normalize_exact_text(candidate) == normalize_exact_text(expected)


def case_hash(case: Mapping[str, Any]) -> str:
    """Hash the stable case identity while ignoring the stored hash field."""

    payload = {key: value for key, value in dict(case).items() if key != "case_hash"}
    return stable_hash(payload)


def normalize_verifier_decision(value: Any) -> str:
    """Parse the Exp 3287 strict leading-token decision contract."""

    text = str(value or "").strip()
    if not text:
        return "abstain"
    first = text.split()[0].strip(" \t\r\n.:,;!?\"'`()[]{}").lower()
    return first if first in {"accept", "reject", "abstain"} else "abstain"


def clean_candidate_answer(value: Any) -> str:
    """Strip common answer wrappers while preserving the candidate content."""

    text = str(value or "").strip()
    if not text:
        return ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    lowered = first_line.lower()
    for prefix in ("answer:", "repaired answer:", "final answer:"):
        if lowered.startswith(prefix):
            return first_line[len(prefix) :].strip().strip("\"'` ")
    return first_line.strip().strip("\"'` ")


def normalize_exact_text(value: str) -> str:
    """Normalize exact string fixtures conservatively for both sides."""

    return " ".join(value.strip().casefold().split())


def normalize_bool_string(value: str) -> str:
    """Map common boolean literals to true/false while preserving unknown text."""

    normalized = normalize_exact_text(value)
    if normalized in {"true", "yes", "1"}:
        return "true"
    if normalized in {"false", "no", "0"}:
        return "false"
    return normalized


def parse_int_string(value: str) -> int | None:
    """Parse an integer answer and reject non-integer text."""

    text = str(value).strip()
    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-"):
        return int(text) if text[1:].isdigit() else None
    return int(text) if text.isdigit() else None


def source_clean(*sources: Mapping[str, Any]) -> bool:
    """Return whether source artifacts are free of adversarial/corrigendum flags."""

    for source in sources:
        if source.get("flagged_adversarial") is True:
            return False
        pending = source.get("corrigendum_pending")
        if isinstance(pending, list) and pending:
            return False
        if isinstance(pending, str) and pending.strip():
            return False
    return True


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the v11 artifact and fail closed on overclaim-prone fields."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    for field in ("headline_repair_panel_ready", "repair_panel_ran", "headline_claim_allowed"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bool")
    for field in (
        "panel_case_count",
        "verified_success_count",
        "false_accept_count",
        "abstention_count",
        "gpu_mem_used_mib",
        "tokens_generated",
    ):
        value = artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    success_rate = artifact.get("repair_success_rate")
    if (
        not isinstance(success_rate, int | float)
        or isinstance(success_rate, bool)
        or not 0.0 <= float(success_rate) <= 1.0
    ):
        raise ValueError("repair_success_rate must be in [0, 1]")
    for field in ("repair_success_ci95", "false_accept_rate_ci95"):
        value = artifact.get(field)
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, int | float) or isinstance(item, bool) for item in value)
        ):
            raise ValueError(f"{field} must be a two-number list")
    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, int | float) or isinstance(duration_s, bool) or duration_s < 0:
        raise ValueError("duration_s must be a non-negative number")
    for field in (
        "model_specs",
        "models_used",
        "missing_model_specs",
        "preconditions_checked",
        "per_family_metrics",
        "candidate_results",
    ):
        expected = Mapping if field in {"model_specs", "per_family_metrics"} else list
        if not isinstance(artifact.get(field), expected):
            raise ValueError(f"{field} has the wrong type")
    checksum = str(artifact.get("reproducibility_checksum") or "")
    if len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a 64-character checksum")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")
    if artifact.get("repair_panel_ran") is True and not artifact.get("models_used"):
        raise ValueError("models_used required when repair_panel_ran=true")
    if artifact.get("headline_claim_allowed") is True:
        if safe_int(artifact.get("panel_case_count")) < MIN_PANEL_CASES:
            raise ValueError("headline_claim_allowed requires panel_case_count>=30")
        if safe_int(artifact.get("false_accept_count")) != 0:
            raise ValueError("headline_claim_allowed requires zero false accepts")
        if artifact.get("provenance_clean") is not True:
            raise ValueError("headline_claim_allowed requires clean provenance")
        if artifact.get("headline_repair_panel_ready") is not True:
            raise ValueError("headline_claim_allowed requires headline_repair_panel_ready")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that separates execution from promotion."""

    return (
        f"complete: repair_panel_ran={str(artifact['repair_panel_ran']).lower()}; "
        f"headline_repair_panel_ready={str(artifact['headline_repair_panel_ready']).lower()}; "
        f"panel_case_count={artifact['panel_case_count']}; "
        f"verified_success_count={artifact['verified_success_count']}; "
        f"false_accept_count={artifact['false_accept_count']}; "
        f"abstention_count={artifact['abstention_count']}; "
        f"headline_claim_allowed={str(artifact['headline_claim_allowed']).lower()}"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable panel inputs while excluding timing and local test-run noise."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum", "tests_run"}
    }
    return stable_hash(stable)


def run_llama_repair_panel(
    cases: list[JsonDict],
    model: JsonDict,
    random_seed: int,
) -> JsonDict:  # pragma: no cover - exercised by live artifact generation.
    """Generate repairs and calibrated verifier decisions with one local GGUF model."""

    from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415

    llm = Llama(
        model_path=str(model["model_path"]),
        n_ctx=2048,
        n_gpu_layers=-1,
        seed=int(random_seed),
        verbose=False,
    )
    grammar = LlamaGrammar.from_string(DECISION_GRAMMAR)
    rows: list[JsonDict] = []
    for case in cases:
        candidate_text, candidate_tokens = llama_chat(
            llm,
            system=(
                "Repair one exact fixture answer. Use only the local context. "
                "Reply with only the repaired answer, no explanation."
            ),
            user=repair_prompt(case),
            max_tokens=32,
        )
        candidate_answer = clean_candidate_answer(candidate_text)
        verifier_text, verifier_tokens = llama_chat(
            llm,
            system=(
                "You verify exact fixture rows. Reply with exactly one word: "
                "ACCEPT, REJECT, or ABSTAIN."
            ),
            user=verifier_prompt(case, candidate_answer),
            max_tokens=4,
            grammar=grammar,
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "candidate_answer": candidate_answer,
                "raw_candidate_output": candidate_text,
                "verifier_output_text": verifier_text,
                "token_counts": merge_token_counts(candidate_tokens, verifier_tokens),
            }
        )
    return {"rows": rows, "gpu_mem_used_mib": gpu_memory_used_mib()}


def llama_chat(
    llm: Any,
    *,
    system: str,
    user: str,
    max_tokens: int,
    grammar: Any = None,
) -> tuple[str, JsonDict]:  # pragma: no cover - llama.cpp boundary.
    """Call llama.cpp chat completion and extract text plus token accounting."""

    kwargs: JsonDict = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
    }
    if grammar is not None:
        kwargs["grammar"] = grammar
    response = llm.create_chat_completion(**kwargs)
    return extract_chat_text(response), mapping(response.get("usage") if isinstance(response, Mapping) else {})


def repair_prompt(case: Mapping[str, Any]) -> str:  # pragma: no cover - live prompt boundary.
    """Prompt text for the repair-generation half of the panel."""

    return (
        f"Context: {case.get('context')}\n"
        f"Question: {case.get('question')}\n"
        f"Failing answer: {case.get('failing_candidate')}\n"
        f"Verifier feedback: {case.get('localized_repair_feedback')}\n"
        "Return only the corrected answer."
    )


def verifier_prompt(case: Mapping[str, Any], candidate_answer: str) -> str:  # pragma: no cover
    """Prompt text for the calibrated clean-verifier half of the panel."""

    return (
        "You are checking one exact fixture row.\n"
        "Reply with exactly one word: ACCEPT, REJECT, or ABSTAIN.\n"
        f"Context: {case.get('context')}\n"
        f"Question: {case.get('question')}\n"
        f"Candidate answer: {candidate_answer}\n"
        "Does the candidate answer exactly satisfy the context and question?"
    )


def extract_chat_text(response: Any) -> str:  # pragma: no cover - llama.cpp boundary.
    """Extract assistant text from llama.cpp's OpenAI-compatible response."""

    choices = response.get("choices") if isinstance(response, Mapping) else []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(first.get("text") or "")


def default_nvidia_smi_probe() -> JsonDict:  # pragma: no cover - hardware boundary.
    """Check visible NVIDIA GPUs before loading a large local GGUF."""

    payload: JsonDict = {"name": "nvidia_smi", "passed": False, "gpu_count": 0, "error": ""}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    payload.update(
        {
            "passed": result.returncode == 0 and bool(rows),
            "returncode": result.returncode,
            "gpu_count": len(rows),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    )
    return payload


def selected_python_cuda_probe() -> JsonDict:  # pragma: no cover - hardware boundary.
    """Check that the selected Python can use CUDA and llama.cpp GPU offload."""

    payload: JsonDict = {
        "name": "selected_python_cuda",
        "selected_python": sys.executable,
        "passed": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "torch_import_ok": False,
        "llama_cpp_import_ok": False,
        "llama_cpp_supports_gpu_offload": False,
        "error": "",
    }
    try:
        import torch  # noqa: PLC0415

        payload["torch_import_ok"] = True
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from llama_cpp import llama_cpp as llama_backend  # noqa: PLC0415

        payload["llama_cpp_import_ok"] = True
        payload["llama_cpp_supports_gpu_offload"] = bool(
            llama_backend.llama_supports_gpu_offload()
        )
    except Exception as exc:
        suffix = f"{type(exc).__name__}: {exc}"
        payload["error"] = suffix if not payload["error"] else payload["error"] + "; " + suffix
    payload["passed"] = (
        payload["cuda_available"] is True
        and safe_int(payload["cuda_device_count"]) > 0
        and payload["llama_cpp_import_ok"] is True
        and payload["llama_cpp_supports_gpu_offload"] is True
    )
    return payload


def gpu_memory_used_mib() -> int:  # pragma: no cover - hardware boundary.
    """Return total visible NVIDIA memory use in MiB after the live run."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    return sum(safe_int(line.strip()) for line in result.stdout.splitlines())


def merge_token_counts(*rows: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - live metadata.
    """Merge llama.cpp usage dictionaries from generation and verification calls."""

    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        for key in merged:
            merged[key] += safe_int(row.get(key))
    return merged


def source_artifact_row(load: JsonLoad, label: str) -> JsonDict:
    """Return compact source-artifact provenance for a JSON input."""

    return {
        "label": label,
        "path": relative_or_abs(REPO_ROOT, load.path),
        "present": load.present,
        "readable": load.readable,
        "error": load.error,
        "sha256": load.sha256,
    }


def file_source_artifact_row(path: Path, label: str) -> JsonDict:
    """Return compact source-artifact provenance for a non-JSON input."""

    return {
        "label": label,
        "path": relative_or_abs(REPO_ROOT, path),
        "present": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def normalize_precondition(payload: Mapping[str, Any]) -> JsonDict:
    """Normalize injected or live precondition probes to a common shape."""

    row = dict(payload)
    row["name"] = str(row.get("name") or "unnamed_precondition")
    row["passed"] = row.get("passed") is True
    return row


def resolve_path(root: Path, value: str) -> Path:
    """Resolve absolute and repository-relative model paths."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def relative_or_abs(root: Path, path: Path) -> str:
    """Render a path relative to root when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate with visible zero-denominator behavior."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def duration(started: float, finished: float) -> float:
    """Return non-negative wall-clock duration rounded for stable JSON."""

    return round(max(0.0, float(finished) - float(started)), 6)


def sha256_file(path: Path) -> str | None:
    """Hash a local file for provenance, returning None when it cannot be read."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def sha256_text(text: str) -> str:
    """Hash serialized text for artifact-to-file integrity checks."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 for JSON-compatible payloads."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def mapping(value: Any) -> JsonDict:
    """Return a plain dict for mapping-like values, otherwise an empty dict."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from an arbitrary sequence."""

    return [dict(item) for item in sequence(value) if isinstance(item, Mapping)]


def sequence(value: Any) -> list[Any]:
    """Return lists and tuples as lists while rejecting strings as sequences."""

    return list(value) if isinstance(value, list | tuple) else []


def safe_int(value: Any, *, default: int = 0) -> int:
    """Convert simple numeric values to int without letting booleans masquerade as counts."""

    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:  # pragma: no cover - CLI wrapper.
    """Write the default Exp 3302 artifact in the repository results directory."""

    output = write_artifact()
    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
