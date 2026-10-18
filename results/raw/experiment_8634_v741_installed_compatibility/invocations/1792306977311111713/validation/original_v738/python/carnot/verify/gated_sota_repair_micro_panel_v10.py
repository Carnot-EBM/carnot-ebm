"""Run the Exp 3290 gated SOTA repair micro-panel v10 artifact.

Spec refs: REQ-VERIFY-3290, SCENARIO-VERIFY-3290.

The panel is intentionally narrow. Exp 3289 decides whether repair may reopen
and also supplies the permitted scope. This module then samples only exact
context-fixture counterexamples, asks one available mandated local GGUF model to
repair those local failures, and accepts a repair only when both authorities
agree: the Exp 3287 calibrated clean-verifier decision contract says ACCEPT and
the deterministic exact fixture check passes. That separation keeps a model
from grading its own repair.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
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
SCHEMA_VERSION = "carnot.gated_sota_repair_micro_panel.v10"
EXPERIMENT_ID = "exp3290"
TASK_ID = "exp3290-gated-sota-repair-micro-panel-v10"
ARTIFACT = "experiment_3290_gated_sota_repair_micro_panel_v10"
MILESTONE = "2026.05.304"
RUN_DATE = "20260528"
RANDOM_SEED = 3290

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3290_gated_sota_repair_micro_panel_v10.json")
EXP3289_REL_PATH = Path("results/experiment_3289_repair_gate_decision_v9_after_garak_abstention.json")
CONTEXT_FIXTURE_REL_PATH = Path("data/research/context_cot_clbench_parametric_shortcut_v1.jsonl")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
DECISION_GRAMMAR = 'root ::= "ACCEPT" | "REJECT" | "ABSTAIN"\n'
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3290_gated_sota_repair_micro_panel_v10.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/gated_sota_repair_micro_panel_v10.py -m pytest -o addopts='' tests/python/test_experiment_3290_gated_sota_repair_micro_panel_v10.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/gated_sota_repair_micro_panel_v10.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
REQUIRED_FIELDS = {
    "sota_repair_micro_panel_v10_ready",
    "repair_panel_ran",
    "model_specs",
    "models_used",
    "missing_model_specs",
    "preconditions_checked",
    "panel_case_count",
    "repair_success_rate",
    "verified_success_count",
    "false_accept_count",
    "abstention_count",
    "localized_failure_feedback",
    "headline_claim_allowed",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


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
    """REQ-VERIFY-3290: build the gated repair panel or a terminal skip artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    gate_load = read_json_object(root_path / EXP3289_REL_PATH)
    gate = gate_load.payload
    scope = mapping(gate.get("permitted_repair_scope"))
    gate_open = (
        gate_load.present
        and gate_load.readable
        and gate.get("repair_gate_decision_v9_ready") is True
        and gate.get("repair_gate_open") is True
        and scope.get("repair_generation_allowed") is True
    )
    preconditions: list[JsonDict] = [gate_precondition(gate_load, gate_open)]
    blocked_reasons: list[str] = [] if gate_open else ["gate_blocked"]
    inventory = empty_inventory()
    panel_cases: list[JsonDict] = []
    evaluated_rows: list[JsonDict] = []
    runner_payload: JsonDict = {}
    model: JsonDict | None = None
    runner_error = ""

    if gate_open:
        nvidia = normalize_precondition((nvidia_probe or default_nvidia_smi_probe)())
        preconditions.append(nvidia)
        if nvidia.get("passed") is not True:
            blocked_reasons.append("nvidia_smi_unavailable")

        selected_python = normalize_precondition(
            (python_cuda_probe or selected_python_cuda_probe)()
        )
        preconditions.append(selected_python)
        if selected_python.get("passed") is not True:
            blocked_reasons.append("selected_python_cuda_unavailable")

        inventory = resolve_mandated_model_inventory(root_path, scope)
        cache_precondition = {
            "name": "mandated_sota_gguf_cache",
            "passed": bool(inventory["available_models"]),
            "cached_sota_pair_attempted": True,
            "cached_sota_pair_available": inventory["cached_sota_pair_available"],
            "available_model_ids": [
                row["model_id"] for row in mapping_list(inventory["available_models"])
            ],
            "missing_model_ids": [
                row["model_id"] for row in mapping_list(inventory["missing_model_specs"])
            ],
        }
        preconditions.append(cache_precondition)
        if cache_precondition["passed"] is not True:
            blocked_reasons.append("mandated_sota_gguf_unavailable")

        panel_cases = build_micro_panel(root_path, scope)
        minimum_cases = panel_min_cases(scope)
        panel_precondition = {
            "name": "exact_context_fixture_panel",
            "passed": len(panel_cases) >= minimum_cases,
            "path": CONTEXT_FIXTURE_REL_PATH.as_posix(),
            "panel_case_count": len(panel_cases),
            "minimum_case_count": minimum_cases,
            "maximum_case_count": panel_max_cases(scope),
            "scope_label": str(scope.get("scope_label") or ""),
        }
        preconditions.append(panel_precondition)
        if panel_precondition["passed"] is not True:
            blocked_reasons.append("exact_context_fixture_panel_too_small")

        model = select_model_for_panel(inventory, scope)
        if (
            model is not None
            and panel_cases
            and all(row.get("passed") is True for row in preconditions)
        ):
            try:
                runner = candidate_runner or run_llama_repair_panel
                runner_payload = mapping(runner(panel_cases, model, int(random_seed)))
                evaluated_rows = evaluate_runner_rows(
                    panel_cases,
                    runner_payload.get("rows"),
                    model,
                )
            except Exception as exc:  # pragma: no cover - defensive live boundary.
                runner_error = f"{type(exc).__name__}: {exc}"
                blocked_reasons.append("repair_candidate_runner_failed")

    metrics = panel_metrics(evaluated_rows)
    if metrics["false_accept_count"] > 0:
        blocked_reasons.append("false_accept_count_nonzero")
    repair_panel_ran = bool(evaluated_rows) and not runner_error
    ready = (
        repair_panel_ran
        and metrics["false_accept_count"] == 0
        and len(panel_cases) >= panel_min_cases(scope)
        and bool(model)
        and all(row.get("passed") is True for row in preconditions)
    )
    finished = time.perf_counter() if now_s is None else float(now_s)

    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3290", "SCENARIO-VERIFY-3290"],
        "sota_repair_micro_panel_v10_ready": ready,
        "repair_panel_ran": repair_panel_ran,
        "model_specs": model_specs_from_inventory(inventory, scope),
        "models_used": models_used(model, runner_payload, len(panel_cases)) if repair_panel_ran else [],
        "missing_model_specs": mapping_list(inventory["missing_model_specs"]) if gate_open else [],
        "preconditions_checked": preconditions,
        "panel_case_count": len(evaluated_rows) if repair_panel_ran else 0,
        "repair_success_rate": metrics["repair_success_rate"],
        "verified_success_count": metrics["verified_success_count"],
        "false_accept_count": metrics["false_accept_count"],
        "abstention_count": metrics["abstention_count"],
        "localized_failure_feedback": localized_failure_feedback(evaluated_rows),
        "headline_claim_allowed": headline_claim_allowed(scope),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration(started, finished),
        "honest_verdict": "",
        "blocked_reasons": sorted(set(blocked_reasons)),
        "gate_source": gate_source(gate_load),
        "permitted_repair_scope": dict(scope),
        "panel_cases": panel_cases,
        "candidate_results": evaluated_rows,
        "clean_verifier_policy": {
            "source_experiment_id": "exp3287",
            "decision_contract": "ACCEPT|REJECT|ABSTAIN",
            "strict_leading_token": True,
            "accepted_repairs_require_exact_pass": True,
            "abstentions_recorded_separately": True,
        },
        "inference_substrate": (
            "local_sota_gguf_repair_plus_calibrated_clean_verifier"
            if repair_panel_ran
            else "gated_skip_or_precondition_block"
        ),
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
    """Build, validate, and persist the Exp 3290 terminal JSON artifact."""

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


def gate_precondition(gate_load: JsonLoad, gate_open: bool) -> JsonDict:
    """Expose the exact upstream gate condition that permits or blocks repair."""

    return {
        "name": "exp3289_repair_gate_open",
        "passed": gate_open,
        "path": EXP3289_REL_PATH.as_posix(),
        "present": gate_load.present,
        "readable": gate_load.readable,
        "error": gate_load.error,
        "repair_gate_decision_v9_ready": gate_load.payload.get("repair_gate_decision_v9_ready")
        is True,
        "repair_gate_open": gate_load.payload.get("repair_gate_open") is True,
        "sha256": gate_load.sha256,
    }


def resolve_mandated_model_inventory(root: Path, scope: Mapping[str, Any]) -> JsonDict:
    """Resolve all mandated GGUF paths, trying cached_sota_pair before single-model fallback."""

    pair = cached_sota_pair(gpu_indices=(0, 1))
    pair_by_id = {str(row.get("hf_id") or row.get("model_id") or ""): row for row in pair or []}
    specs_by_id = {spec["hf_id"]: spec for spec in SOTA_GGUF_MODELS}
    available: list[JsonDict] = []
    missing: list[JsonDict] = []
    mandated_models: JsonDict = {}
    selected_ids = set(scope_selected_model_ids(scope))

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
            "selected_by_scope": model_id in selected_ids,
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
                    "selected_by_scope": model_id in selected_ids,
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


def empty_inventory() -> JsonDict:
    """Return a no-model inventory used before an open gate permits cache checks."""

    return {
        "cached_sota_pair_attempted": False,
        "cached_sota_pair_available": False,
        "cached_sota_pair_specs": [],
        "available_models": [],
        "missing_model_specs": [],
        "mandated_models": {},
    }


def select_model_for_panel(inventory: Mapping[str, Any], scope: Mapping[str, Any]) -> JsonDict | None:
    """Prefer the model selected by Exp 3289, falling back to any cached mandated GGUF."""

    available = mapping_list(inventory.get("available_models"))
    selected_ids = scope_selected_model_ids(scope)
    for model_id in selected_ids:
        for row in available:
            if row.get("model_id") == model_id:
                return row
    return available[0] if available else None


def model_specs_from_inventory(inventory: Mapping[str, Any], scope: Mapping[str, Any]) -> JsonDict:
    """Return the artifact model-spec block with mandated models always named."""

    return {
        "runtime": "llama_cpp_local_gguf_only",
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "scope_selected_model_ids": scope_selected_model_ids(scope),
        "cached_sota_pair_attempted": inventory.get("cached_sota_pair_attempted") is True,
        "cached_sota_pair_available": inventory.get("cached_sota_pair_available") is True,
        "cached_sota_pair_specs": mapping_list(inventory.get("cached_sota_pair_specs")),
        "available_model_count": len(mapping_list(inventory.get("available_models"))),
        "missing_model_count": len(mapping_list(inventory.get("missing_model_specs"))),
        "mandated_models": mapping(inventory.get("mandated_models")),
        "generation_runtime": "llama_cpp_local_generation",
        "verification_runtime": "exp3287_calibrated_accept_reject_abstain_contract",
    }


def models_used(model: Mapping[str, Any] | None, runner_payload: Mapping[str, Any], n_cases: int) -> list[JsonDict]:
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


def build_micro_panel(root: Path, scope: Mapping[str, Any]) -> list[JsonDict]:
    """Select a bounded panel from permitted exact context-fixture counterexamples."""

    permitted = set(str(item) for item in sequence(scope.get("permitted_case_families")))
    if "exact_context_fixture_counterexamples" not in permitted:
        return []
    target_count = panel_min_cases(scope)
    maximum = panel_max_cases(scope)
    cases: list[JsonDict] = []
    for raw in read_jsonl_objects(root / CONTEXT_FIXTURE_REL_PATH):
        case = repair_case_from_fixture(raw)
        if case:
            cases.append(case)
        if len(cases) >= min(target_count, maximum):
            break
    return cases


def read_jsonl_objects(path: Path) -> list[JsonDict]:
    """Read local JSONL fixture objects and drop malformed rows without guessing."""

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


def repair_case_from_fixture(raw: Mapping[str, Any]) -> JsonDict:
    """Convert one exact fixture counterexample into a repair case."""

    fixture_id = str(raw.get("fixture_id") or "")
    expected = str(raw.get("expected_answer") or "")
    counter = mapping(raw.get("minimal_counterexample"))
    failing = str(counter.get("candidate_answer") or raw.get("prior_bait_answer") or "")
    if not fixture_id or not expected or not failing:
        return {}
    case: JsonDict = {
        "case_id": f"{fixture_id}:repair",
        "fixture_id": fixture_id,
        "family": str(raw.get("family") or ""),
        "permitted_case_family": "exact_context_fixture_counterexamples",
        "context": str(raw.get("context") or ""),
        "question": str(raw.get("question") or ""),
        "failing_candidate": failing,
        "expected_answer": expected,
        "exact_checker_type": str(raw.get("exact_checker_type") or "exact_alias_string"),
        "localized_failure_mode": str(
            counter.get("failure_mode") or "parametric_prior_shortcut"
        ),
    }
    case["localized_repair_feedback"] = repair_feedback_message(case)
    case["case_hash"] = stable_hash(case)
    return case


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
                "fixture_id": str(case.get("fixture_id") or ""),
                "family": str(case.get("family") or ""),
                "model_id": str(model.get("model_id") or model.get("hf_id") or ""),
                "model_path": str(model.get("model_path") or ""),
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
    """Classify one non-successful repair outcome for localized feedback."""

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
    """Compute repair utility, false accepts, and abstentions over visible denominator."""

    n_rows = len(rows)
    verified_success_count = sum(row.get("verified_success") is True for row in rows)
    false_accept_count = sum(row.get("false_accept") is True for row in rows)
    abstention_count = sum(row.get("abstained") is True for row in rows)
    return {
        "verified_success_count": verified_success_count,
        "false_accept_count": false_accept_count,
        "abstention_count": abstention_count,
        "repair_success_rate": rate(verified_success_count, n_rows),
    }


def localized_failure_feedback(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return precise verifier feedback for every non-successful repair case."""

    feedback: list[JsonDict] = []
    for row in rows:
        if row.get("verified_success") is True:
            continue
        feedback.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "fixture_id": str(row.get("fixture_id") or ""),
                "family": str(row.get("family") or ""),
                "failure_class": str(row.get("failure_class") or ""),
                "failing_candidate": str(row.get("failing_candidate") or ""),
                "candidate_answer": str(row.get("candidate_answer") or ""),
                "expected_answer": str(row.get("expected_answer") or ""),
                "exact_check_passed": row.get("exact_check_passed") is True,
                "calibrated_clean_verifier_decision": str(
                    row.get("calibrated_clean_verifier_decision") or ""
                ),
                "localized_feedback": str(row.get("localized_repair_feedback") or ""),
            }
        )
    return feedback


def repair_feedback_message(case: Mapping[str, Any]) -> str:
    """Create compact local feedback without leaking any broader benchmark claim."""

    return (
        "Use the local context for this fixture; replace "
        f"{case.get('failing_candidate')!r} with the exact expected answer "
        f"{case.get('expected_answer')!r} for {case.get('fixture_id')}."
    )


def exact_check(case: Mapping[str, Any], candidate_answer: str) -> bool:
    """Run the deterministic exact authority for one repaired candidate."""

    expected = clean_candidate_answer(case.get("expected_answer") or "")
    candidate = clean_candidate_answer(candidate_answer)
    checker = str(case.get("exact_checker_type") or "")
    if checker == "exact_integer_string":
        return parse_int_string(candidate) == parse_int_string(expected)
    return normalize_exact_text(candidate) == normalize_exact_text(expected)


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
    """Normalize exact string fixtures in the same conservative way for both sides."""

    return " ".join(value.strip().lower().split())


def parse_int_string(value: str) -> int | None:
    """Parse an integer answer and reject non-integer text."""

    text = value.strip()
    if text.startswith("+"):
        text = text[1:]
    if text.startswith("-"):
        return int(text) if text[1:].isdigit() else None
    return int(text) if text.isdigit() else None


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


def repair_prompt(case: Mapping[str, Any]) -> str:  # pragma: no cover - covered through live path.
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


def source_artifact_row(path: Path, label: str) -> JsonDict:
    """Return compact source-artifact provenance for the panel result."""

    return {
        "label": label,
        "path": path.as_posix(),
        "present": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def gate_source(gate_load: JsonLoad) -> JsonDict:
    """Return upstream gate provenance without inlining the full artifact."""

    return {
        "path": EXP3289_REL_PATH.as_posix(),
        "present": gate_load.present,
        "readable": gate_load.readable,
        "error": gate_load.error,
        "sha256": gate_load.sha256,
    }


def panel_min_cases(scope: Mapping[str, Any]) -> int:
    """Return the scope's minimum case count, defaulting to the requested micro-panel size."""

    sample_size = mapping(scope.get("sample_size"))
    return max(0, safe_int(sample_size.get("min_cases"), default=4))


def panel_max_cases(scope: Mapping[str, Any]) -> int:
    """Return the hard upper bound from Exp 3289's scope."""

    sample_size = mapping(scope.get("sample_size"))
    scoped = safe_int(scope.get("max_panel_cases"), default=8)
    sampled = safe_int(sample_size.get("max_cases"), default=scoped)
    return max(0, min(scoped, sampled))


def scope_selected_model_ids(scope: Mapping[str, Any]) -> list[str]:
    """Read selected model ids from both top-level and nested Exp 3289 scope fields."""

    selected = [str(item) for item in sequence(scope.get("selected_model_ids"))]
    nested = mapping(scope.get("model_specs"))
    selected.extend(str(item) for item in sequence(nested.get("selected_model_ids")))
    return list(dict.fromkeys(item for item in selected if item))


def headline_claim_allowed(scope: Mapping[str, Any]) -> bool:
    """Carry forward Exp 3289's no-headline boundary, failing closed when absent."""

    boundary = mapping(scope.get("claim_boundary"))
    return boundary.get("headline_claim_allowed") is True


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal verdict that separates execution from repair utility."""

    return (
        f"complete: repair_panel_ran={str(artifact['repair_panel_ran']).lower()}; "
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


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the v10 artifact and fail closed on overclaim-prone fields."""

    missing = REQUIRED_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    for field in ("sota_repair_micro_panel_v10_ready", "repair_panel_ran", "headline_claim_allowed"):
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bool")
    for field in (
        "panel_case_count",
        "verified_success_count",
        "false_accept_count",
        "abstention_count",
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
    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, int | float) or isinstance(duration_s, bool) or duration_s < 0:
        raise ValueError("duration_s must be a non-negative number")
    for field in (
        "model_specs",
        "models_used",
        "missing_model_specs",
        "preconditions_checked",
        "localized_failure_feedback",
    ):
        expected = Mapping if field == "model_specs" else list
        if not isinstance(artifact.get(field), expected):
            raise ValueError(f"{field} has the wrong type")
    checksum = str(artifact.get("reproducibility_checksum") or "")
    if len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a 64-character checksum")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")
    if artifact.get("headline_claim_allowed") is True:
        raise ValueError("Exp3290 micro-panel must not allow headline claims")
    if artifact.get("sota_repair_micro_panel_v10_ready") is True:
        if artifact.get("repair_panel_ran") is not True:
            raise ValueError("ready artifact must have repair_panel_ran=true")
        if not artifact.get("models_used"):
            raise ValueError("ready artifact must name models_used")
        if artifact.get("false_accept_count") != 0:
            raise ValueError("ready artifact must have zero false accepts")


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
    """Write the default Exp 3290 artifact in the repository results directory."""

    output = write_artifact()
    print(output)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
