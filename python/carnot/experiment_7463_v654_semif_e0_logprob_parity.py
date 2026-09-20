"""Measure native and local-server option-logit parity for SEMIF E0.

The exact scored wheel is a separate branch. Host packages do not satisfy that
branch because the scored kernel can mount different bytes and expose a
different request schema.

Spec refs: REQ-ARC-WMTE-7463 and SCENARIO-ARC-WMTE-7463-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.request
import zipfile

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.654"
EXPERIMENT_ID = "exp7463-v654-semif-e0-logprob-parity"
SCHEMA = "carnot.exp7463.v654.semif_e0_logprob_parity.v1"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_HF_ID]
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7463_v654_semif_e0_logprob_parity.json")
RAW_DIR = Path("results/raw/experiment_7463_v654_semif_e0_logprob_parity")
MODULE_PATH = Path("python/carnot/experiment_7463_v654_semif_e0_logprob_parity.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7463_v654_semif_e0_logprob_parity.py")
TEST_PATH = Path("tests/python/test_experiment_7463_v654_semif_e0_logprob_parity.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
KERNEL_MANIFEST_PATH = Path("scripts/kaggle/submission_kernel/kernel-metadata.json")
OWNERSHIP_RESULT_PATH = Path("results/experiment_7422_v651_runtime_ownership.json")
CAPTURE_RESULT_PATH = Path("results/experiment_7448_v653_capture_lifecycle.json")
WORLD_MODEL_PATH = Path("python/carnot/agentic/arc_executable_world_model.py")
LLAMA_SERVER_PATH = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
MAX_LIVE_SECONDS = 900.0
MAX_SERVER_REQUESTS = 160
N_DEVELOPMENT = 64

AFFECTED_CHECK_NAMES = REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so prompt or row changes cannot look equivalent."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def utc_now() -> str:
    """Record UTC boundaries while elapsed time uses the monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush every boundary so long model and subprocess work stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7463] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _prompt(question: str, option_a: str, option_b: str) -> str:
    """Use one plain template so both runtimes receive the same prompt bytes."""

    return (
        "Choose the correct answer.\n"
        f"Question: {question}\n"
        f"A: {option_a}\n"
        f"B: {option_b}\n"
        "Return only A or B.\nAnswer:"
    )


def _development_prompt_rows() -> list[JsonDict]:
    """Build the frozen balanced panel without reading model outcomes."""

    rows: list[JsonDict] = []
    for index in range(N_DEVELOPMENT):
        left = 17 + index
        right = 3 + (index % 11)
        correct = left + right
        distractor = correct + 2 + (index % 5)
        expected = "A" if index % 2 == 0 else "B"
        option_a, option_b = (
            (str(correct), str(distractor)) if expected == "A" else (str(distractor), str(correct))
        )
        prompt_text = _prompt(f"What is {left} + {right}?", option_a, option_b)
        rows.append(
            {
                "unit_id": f"development-{index:02d}",
                "prompt": prompt_text,
                "prompt_sha256": canonical_hash(prompt_text),
                "expected_label": expected,
                "correct_value": correct,
                "distractor_value": distractor,
            }
        )
    return rows


_DEVELOPMENT_PROMPTS = _development_prompt_rows()
DEVELOPMENT_PROMPTS_SHA256 = canonical_hash(_DEVELOPMENT_PROMPTS)


def freeze_development_prompts() -> list[JsonDict]:
    """Return a copy so callers cannot mutate the registered panel in memory."""

    return deepcopy(_DEVELOPMENT_PROMPTS)


def freeze_positive_controls() -> list[JsonDict]:
    """Create four known answers in both label orders before model loading."""

    cases = (
        ("Which item is a fruit?", "apple", "hammer"),
        ("What is 2 + 2?", "4", "9"),
        ("Which city is the capital of France?", "Paris", "Cairo"),
        ("Which is liquid at room temperature?", "water", "iron"),
    )
    rows: list[JsonDict] = []
    for index, (question, correct, wrong) in enumerate(cases):
        for order, option_a, option_b, expected in (
            ("original", correct, wrong, "A"),
            ("reversed", wrong, correct, "B"),
        ):
            text = _prompt(question, option_a, option_b)
            rows.append(
                {
                    "unit_id": f"control-{index}-{order}",
                    "order": order,
                    "prompt": text,
                    "prompt_sha256": canonical_hash(text),
                    "expected_label": expected,
                }
            )
    return rows


def option_distribution(option_logits: Mapping[str, float]) -> dict[str, float]:
    """Normalize only two observed logits; absence is never treated as zero."""

    if set(option_logits) != {"A", "B"}:
        raise ValueError("missing_option_logits")
    values = {label: float(option_logits[label]) for label in ("A", "B")}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("nonfinite_option_logit")
    maximum = max(values.values())
    weights = {label: math.exp(value - maximum) for label, value in values.items()}
    total = sum(weights.values())
    return {label: weights[label] / total for label in ("A", "B")}


def _lower_confidence_bound(successes: int, total: int) -> float:
    """Return the exact one-sided 95% Clopper-Pearson lower bound."""

    if total <= 0 or successes <= 0:
        return 0.0
    if successes >= total:
        return 0.05 ** (1.0 / total)
    from scipy.stats import beta  # noqa: PLC0415

    return float(beta.ppf(0.05, successes, total - successes + 1))


def _valid_pair(row: Mapping[str, Any]) -> bool:
    """Require complete options and identical runtime inputs before comparison."""

    native = row.get("native")
    server = row.get("local_server")
    if not isinstance(native, Mapping) or not isinstance(server, Mapping):
        return False
    for runtime in (native, server):
        logits = runtime.get("option_logits")
        probabilities = runtime.get("probabilities")
        if not isinstance(logits, Mapping) or set(logits) != {"A", "B"}:
            return False
        if not isinstance(probabilities, Mapping) or set(probabilities) != {"A", "B"}:
            return False
    return bool(
        row.get("disposition") == "complete"
        and row.get("input_ids_equal") is True
        and row.get("weights_equal") is True
        and row.get("quantization_equal") is True
        and row.get("tokenizer_equal") is True
        and row.get("logit_bias") is None
        and int(row.get("emitted_tokens", 0)) <= 1
    )


def reduce_parity_rows(rows: Sequence[Mapping[str, Any]], *, expected_units: int) -> JsonDict:
    """Recompute agreement and total variation from eligible raw pairs."""

    candidates = [row for row in rows if row.get("row_kind") == "parity_pair"]
    valid = [row for row in candidates if _valid_pair(row)]
    agreements = 0
    distances: list[float] = []
    for row in valid:
        native = row["native"]["probabilities"]
        server = row["local_server"]["probabilities"]
        native_argmax = max(("A", "B"), key=lambda label: float(native[label]))
        server_argmax = max(("A", "B"), key=lambda label: float(server[label]))
        agreements += int(native_argmax == server_argmax)
        distances.append(
            0.5 * sum(abs(float(native[label]) - float(server[label])) for label in ("A", "B"))
        )
    complete = len(valid)
    lower = _lower_confidence_bound(agreements, complete)
    median_tv = statistics.median(distances) if distances else None
    all_present = complete == expected_units == len(candidates)
    identity_equal = all(
        row.get("weights_equal") is True
        and row.get("quantization_equal") is True
        and row.get("tokenizer_equal") is True
        and row.get("input_ids_equal") is True
        for row in valid
    )
    passed = bool(
        all_present
        and identity_equal
        and lower >= 0.95
        and median_tv is not None
        and median_tv <= 0.05
    )
    return {
        "planned": expected_units,
        "complete": complete,
        "unavailable": expected_units - complete,
        "argmax_agreements": agreements,
        "argmax_agreement": agreements / complete if complete else None,
        "argmax_agreement_95_lower": lower,
        "median_tv_distance": median_tv,
        "all_options_present": all_present,
        "runtime_identity_equal": identity_equal,
        "passed": passed,
    }


def _wheel_metadata(wheel: Path) -> JsonDict:
    """Read version metadata from the exact wheel archive, not host imports."""

    metadata_text = ""
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if names:
            metadata_text = archive.read(sorted(names)[0]).decode("utf-8", errors="replace")
    version = next(
        (
            line.split(":", 1)[1].strip()
            for line in metadata_text.splitlines()
            if line.lower().startswith("version:")
        ),
        None,
    )
    return {
        "path": str(wheel),
        "sha256": sha256_file(wheel),
        "size_bytes": wheel.stat().st_size,
        "version": version,
        "metadata_sha256": canonical_hash(metadata_text) if metadata_text else None,
    }


def inspect_scored_runtime(
    manifest_path: Path,
    mounted_root: Path,
    *,
    host_metadata: Path | None = None,
) -> JsonDict:
    """Authenticate exact mounted wheel bytes or return one precise blocker."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = list(manifest.get("dataset_sources") or [])
    declared = "iancblenke/carnot-vllm-wheels-py312" in sources
    wheels = (
        sorted(
            path
            for path in mounted_root.rglob("*.whl")
            if "vllm" in path.name.lower() and path.is_file()
        )
        if mounted_root.is_dir()
        else []
    )
    host_receipt = None
    if host_metadata is not None and host_metadata.is_file():
        host_receipt = {
            "path": str(host_metadata),
            "sha256": sha256_file(host_metadata),
            "evidence_scope": "ordinary_host_install_not_scored_wheel_evidence",
        }
    proposed = {
        "endpoint": "/v1/completions",
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 20,
        "logit_bias": None,
        "preserve_response_field": "choices[0].logprobs.top_logprobs[0]",
        "preserve_usage_field": "usage.prompt_tokens",
        "implementation_scope": "isolated_wrapper_patch_not_applied",
    }
    if not declared:
        failed = "scored_manifest_vllm_dataset_source"
    elif not wheels:
        failed = "exact_mounted_vllm_wheel_bytes"
    else:
        failed = None
    return {
        "available": failed is None,
        "failed_check": failed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_source_declared": declared,
        "expected_mount_glob": str(mounted_root / "**/vllm*.whl"),
        "observed": [str(path) for path in wheels],
        "wheel": _wheel_metadata(wheels[0]) if wheels else None,
        "host_wheel_metadata": host_receipt,
        "host_wheel_is_evidence": False,
        "request_schema_support": "unverified_without_exact_mounted_wheel",
        "proposed_request_change": proposed,
    }


def derive_scores(
    controls: Sequence[Mapping[str, Any]],
    local_reduction: Mapping[str, Any],
    scored_reduction: Mapping[str, Any],
) -> dict[str, int]:
    """Keep native, local, and exact-scored conclusions independent."""

    native_ready = len(controls) == 8 and all(
        row.get("correct") is True
        and row.get("nonuniform") is True
        and row.get("offload_observed") is True
        for row in controls
    )
    return {
        "native_readout_ready_score": int(native_ready),
        "local_runtime_parity_score": int(native_ready and local_reduction.get("passed") is True),
        "scored_runtime_parity_score": int(scored_reduction.get("passed") is True),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    artifact_field: str,
) -> JsonDict:
    """Retain both operands so a failed gate names its exact evidence."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "artifact_field": artifact_field,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first failed check without hiding the remaining failures."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "first_failure": failed[0] if failed else None,
        "failed_checks": [row.get("check") for row in failed],
    }


def _base_gates(
    controls: Sequence[Mapping[str, Any]],
    local: Mapping[str, Any],
    scored_runtime: Mapping[str, Any],
) -> list[JsonDict]:
    """Separate native validity, local parity, and scored availability gates."""

    native = derive_scores(controls, local, {"passed": False})["native_readout_ready_score"]
    return [
        _gate(
            "native_positive_controls",
            "validity",
            1,
            native,
            "==",
            native == 1,
            "Eight correct nonuniform CUDA-backed controls establish a usable native readout.",
            upstream="current_native_llama_cpp",
            artifact_field="native_readout_ready_score",
        ),
        _gate(
            "local_runtime_parity",
            "exploratory_benefit",
            True,
            local.get("passed"),
            "is",
            local.get("passed") is True,
            "Every identical-runtime pair must meet the frozen agreement and TV bounds.",
            upstream="current_local_llama_server",
            artifact_field="local_runtime_reduction.passed",
        ),
        _gate(
            "exact_scored_runtime_available",
            "external_precondition",
            True,
            scored_runtime.get("available"),
            "is",
            scored_runtime.get("available") is True,
            "Only exact mounted wheel bytes can establish scored-runtime parity.",
            upstream=str(scored_runtime.get("manifest_path")),
            artifact_field=str(scored_runtime.get("failed_check")),
        ),
    ]


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain every top-level field so readers know what it can establish."""

    required = {
        "schema": "Use a versioned schema with exact experiment identity, milestone, and terminal status.",
        "run_date": "Use 20260920 and measured UTC and monotonic boundaries with clock identity.",
        "preconditions_checked": "Record exact paths, ownership, device identity, and prerequisites before work.",
        "MODEL_SPECS": "Name the one current Qwen3.8 GGUF required by this LLM task.",
        "model_specs": "Repeat the model list for lowercase-field readers.",
        "model_invoked": "Distinguish current attempted calls from archived and scripted events.",
        "invocation_counts": "Balance attempted, completed, failed, cancelled, and in-flight work.",
        "inference_substrate": "Name the actual native readout and bounded local-server generation.",
        "inference_substrate_class": "Use model_bounded_generation because HTTP emits at most one token.",
        "execution_venue": "Use host and separate CPU, CUDA, and historical board evidence.",
        "duration_s": "Measure current load, forward, generation, numeric, and validation time without padding.",
        "phase_spans": "Bind progress events, monotonic timings, and completed-unit checkpoints.",
        "random_seed": "Freeze ordering and audit seeds while deterministic reductions need no fitting seed.",
        "reproducibility_checksum": "Bind code, protocol, prompts, model, rows, and validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes and flags without rehabilitating history.",
        "rows": "Keep every control and prompt pair, including unavailable scored-runtime work.",
        "sample_size_budget": "Separate planned, attempted, complete, failed, censored, and unstarted units.",
        "acceptance_gate_results": "Keep gate category, operands, operator, result, and principle explicit.",
        "gate_check_summary": "Name the first exact failed upstream field without hiding later failures.",
        "honest_verdict": "Use a terminal complete prefix while stating the exact blocked runtime branch.",
        "verdict_class": "Use the closed verdict enum and reserve blocked for unchanged external absence.",
        "verifier_is_oracle": "False because this experiment measures runtime agreement, not semantic truth.",
        "flagged_adversarial": "Retain structural reader findings and never clear them to open a gate.",
        "validation_receipts": "Capture exact affected commands, exits, log hashes, and required status.",
        "field_principles": "Echo why each field and gate exists for audit.",
        "native_readout_ready_score": "Bare 0/1 from current owned native positive controls only.",
        "local_runtime_parity_score": "Bare 0/1 for identical-weight local runtimes only.",
        "scored_runtime_parity_score": "Bare 0/1 that only exact attached-runtime evidence can set.",
        "runtime_manifest": "Pin wheel, tokenizer, prompt, quantization, and device per comparison.",
        "proposed_request_change": "Describe the isolated request and preservation patch without applying it.",
    }
    return {
        field: required.get(field, f"Retain {field} as auditable experiment evidence.")
        for field in fields
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the artifact while excluding the checksum and explanatory map itself."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "field_principles"}
    }
    return canonical_hash(payload)


def _zero_counts() -> JsonDict:
    """Return a complete counter shape for fixture and blocked paths."""

    return {
        f"{operation}_{state}": 0
        for operation in ("model_loads", "forward_calls", "generation_calls")
        for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }


def build_artifact_for_test(
    rows: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build a deterministic terminal fixture through production reducers."""

    local = reduce_parity_rows(rows, expected_units=N_DEVELOPMENT)
    scored_runtime = {
        "available": False,
        "failed_check": "exact_mounted_vllm_wheel_bytes",
        "manifest_path": KERNEL_MANIFEST_PATH.as_posix(),
    }
    gates = _base_gates(controls, local, scored_runtime)
    scores = derive_scores(controls, local, {"passed": False})
    counts = _zero_counts()
    counts.update(
        {
            "model_loads_attempted": 2,
            "model_loads_completed": 2,
            "forward_calls_attempted": 72,
            "forward_calls_completed": 72,
            "generation_calls_attempted": 64,
            "generation_calls_completed": 64,
        }
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "complete_scored_runtime_unavailable",
        "run_date": RUN_DATE,
        "started_at_utc": "2026-09-20T00:00:00Z",
        "completed_at_utc": "2026-09-20T00:00:01Z",
        "monotonic_clock": "time.monotonic_ns",
        "started_monotonic_ns": 1,
        "ended_monotonic_ns": 1_000_000_001,
        "preconditions_checked": [],
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": MODEL_SPECS,
        "model_invoked": True,
        "invocation_counts": counts,
        "inference_substrate": "native_llama_cpp_logits_plus_owned_llama_server_one_token",
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": "host",
        "duration_s": 1.0,
        "duration_breakdown_s": {
            "model_load": 0.4,
            "forward": 0.2,
            "generation": 0.2,
            "numeric_reduction": 0.1,
            "validation": 0.1,
        },
        "phase_spans": [],
        "random_seed": {"ordering": 7463, "audit": 7464, "bootstrap": None, "fit": None},
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [deepcopy(dict(row)) for row in rows],
        "positive_control_rows": [deepcopy(dict(row)) for row in controls],
        "sample_size_budget": {
            "planned_independent_units": 72,
            "attempted_independent_units": 72,
            "complete_independent_units": 72,
            "failed_independent_units": 0,
            "censored_independent_units": 0,
            "unstarted_independent_units": 64,
            "server_requests_planned": 64,
            "server_requests_attempted": 64,
            "server_request_cap": MAX_SERVER_REQUESTS,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": "complete_scored_runtime_unavailable_local_parity_measured",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": {},
        **scores,
        "local_runtime_reduction": local,
        "scored_runtime_reduction": {"passed": False, "status": "blocked_no_run"},
        "runtime_manifest": {"scored_vllm": scored_runtime},
        "proposed_request_change": inspect_scored_runtime.__doc__,
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "pending",
            "numbered_e2e": "not_applicable_isolated_readout_experiment",
        },
        "deployment_authorized": False,
        "scored_path_changed": False,
        "production_defaults_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _receipt_errors(receipts: Sequence[Mapping[str, Any]], *, terminal: bool) -> list[str]:
    """Require one successful receipt for every frozen command name."""

    required = set(AFFECTED_CHECK_NAMES)
    if terminal:
        required.update(TERMINAL_CHECK_NAMES)
    errors: list[str] = []
    for name in sorted(required):
        matches = [row for row in receipts if row.get("name") == name]
        if len(matches) != 1:
            errors.append(f"receipt_count:{name}:{len(matches)}")
        elif not (
            matches[0].get("passed") is True
            and matches[0].get("exit_code") == 0
            and matches[0].get("timed_out") is not True
        ):
            errors.append(f"receipt_failed:{name}")
    return errors


def finalize_validation(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Attach required receipts without converting external absence into success."""

    result = deepcopy(dict(artifact))
    receipt_rows = [deepcopy(dict(row)) for row in receipts]
    errors = _receipt_errors(receipt_rows, terminal=True)
    gates = [
        deepcopy(dict(row))
        for row in result["acceptance_gate_results"]
        if row.get("check") != "required_validation"
    ]
    gates.append(
        _gate(
            "required_validation",
            "validity",
            [],
            errors,
            "==",
            not errors,
            "Every affected check and terminal reader must pass once.",
            upstream="validation_receipts",
            artifact_field="validation_receipts",
        )
    )
    if errors:
        result.update(
            {
                "status": "complete_required_validation_failed",
                "honest_verdict": "complete_disqualified_required_validation_failed",
                "verdict_class": "disqualified",
                "native_readout_ready_score": 0,
                "local_runtime_parity_score": 0,
                "scored_runtime_parity_score": 0,
            }
        )
    else:
        result["capability_e2e"]["fresh_process_cold_replay"] = "passed"
    result["flagged_adversarial"] = any(
        row.get("name") == "adversarial_verify" and row.get("passed") is not True
        for row in receipt_rows
    )
    result["validation_receipts"] = receipt_rows
    result["acceptance_gate_results"] = gates
    result["gate_check_summary"] = _gate_summary(gates)
    result["field_principles"] = _field_principles(tuple(result))
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, raw reductions, scores, gates, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    expected_identity = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": MODEL_SPECS,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": "host",
        "verifier_is_oracle": False,
        "deployment_authorized": False,
        "scored_path_changed": False,
    }
    for field, expected in expected_identity.items():
        if artifact.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    rows = artifact.get("rows")
    row_values = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    reduced = reduce_parity_rows(row_values, expected_units=N_DEVELOPMENT)
    if artifact.get("local_runtime_reduction") != reduced:
        errors.append("local_reduction_mismatch")
    controls = artifact.get("positive_control_rows")
    control_values = (
        [row for row in controls if isinstance(row, Mapping)] if isinstance(controls, list) else []
    )
    scored = artifact.get("scored_runtime_reduction")
    scored_value = scored if isinstance(scored, Mapping) else {"passed": False}
    expected_scores = derive_scores(control_values, reduced, scored_value)
    validation_failed = artifact.get("verdict_class") == "disqualified"
    for field, expected in expected_scores.items():
        if validation_failed:
            expected = 0
        if artifact.get(field) != expected:
            errors.append(f"score_mismatch:{field}")
    receipts = artifact.get("validation_receipts")
    receipt_values = (
        [row for row in receipts if isinstance(row, Mapping)] if isinstance(receipts, list) else []
    )
    if require_validation:
        errors.extend(_receipt_errors(receipt_values, terminal=True))
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or artifact.get("gate_check_summary") != _gate_summary(gates):
        errors.append("gate_summary_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Build the fixed Exp7358/Exp7303 affected-file validation plan."""

    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if errors:
        raise ValueError("invalid_validation_plan:" + ",".join(errors))
    return list(commands)


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build the entrypoint replay, independent reducer, and strict readers."""

    python = str(root / ".venv/bin/python")
    relative = candidate.relative_to(root).as_posix()
    reducer = (
        "import json,sys; from pathlib import Path; "
        "from carnot import experiment_7463_v654_semif_e0_logprob_parity as m; "
        "v=json.loads(Path(sys.argv[1]).read_text()); "
        "e=m.validate_artifact(v,require_validation=False); "
        "print(json.dumps({'errors':e},sort_keys=True),flush=True); raise SystemExit(bool(e))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--validate", relative),
            "capability_e2e",
            180,
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, relative),
            "completion",
            180,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "completion",
            180,
        ),
    ]


def _load_json(path: Path) -> JsonDict:
    """Load one required object and keep malformed inputs visibly empty."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _check_row(
    check: str, path: Path, expected: Any, observed: Any, *, artifact_field: str
) -> JsonDict:
    """Record one precondition comparison before dependent model work."""

    return {
        "check": check,
        "path": str(path),
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _gpu_inventory() -> list[JsonDict]:  # pragma: no cover - live host inventory.
    """Read device identity and capacity without changing other users' processes."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    rows: list[JsonDict] = []
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_used_mb": int(parts[3]),
                "memory_free_mb": int(parts[4]),
                "utilization_pct": int(parts[5]),
            }
        )
    return rows


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - live host inputs.
    """Authenticate historical flags, model bytes, runtime, and scored manifest."""

    ownership = _load_json(root / OWNERSHIP_RESULT_PATH)
    capture = _load_json(root / CAPTURE_RESULT_PATH)
    model = cached_current_model(gpu_index=0)
    model_path = Path(str(model.get("model_path"))) if model else Path("/absent-model")
    gpus = _gpu_inventory()
    idle = [
        row
        for row in gpus
        if row["memory_free_mb"] >= 20_000
        and row["memory_used_mb"] <= 1_024
        and row["utilization_pct"] <= 5
    ]
    scored = inspect_scored_runtime(
        root / KERNEL_MANIFEST_PATH,
        Path("/kaggle/input"),
        host_metadata=root
        / ".venv-vllm-trial/lib/python3.12/site-packages/vllm-0.29.0.dist-info/METADATA",
    )
    checks = [
        _check_row(
            "force_live",
            Path("environment:CARNOT_FORCE_LIVE"),
            "1",
            os.environ.get("CARNOT_FORCE_LIVE"),
            artifact_field="CARNOT_FORCE_LIVE",
        ),
        _check_row(
            "ownership_ready",
            root / OWNERSHIP_RESULT_PATH,
            1,
            ownership.get("runtime_ownership_ready_score"),
            artifact_field="runtime_ownership_ready_score",
        ),
        _check_row(
            "ownership_unflagged",
            root / OWNERSHIP_RESULT_PATH,
            False,
            ownership.get("flagged_adversarial"),
            artifact_field="flagged_adversarial",
        ),
        _check_row(
            "capture_ready",
            root / CAPTURE_RESULT_PATH,
            1,
            capture.get("capture_lifecycle_ready_score"),
            artifact_field="capture_lifecycle_ready_score",
        ),
        _check_row(
            "capture_unflagged",
            root / CAPTURE_RESULT_PATH,
            False,
            capture.get("flagged_adversarial"),
            artifact_field="flagged_adversarial",
        ),
        _check_row(
            "model_identity",
            model_path,
            MODEL_HF_ID,
            model.get("hf_id") if model else None,
            artifact_field="hf_id",
        ),
        _check_row(
            "cached_gguf",
            model_path,
            True,
            model_path.is_file(),
            artifact_field="model_path",
        ),
        _check_row(
            "embedded_tokenizer_representation",
            model_path,
            ".gguf",
            model_path.suffix.lower(),
            artifact_field="model_representation",
        ),
        _check_row(
            "cuda_device",
            Path("nvidia-smi"),
            True,
            bool(idle),
            artifact_field="idle_owned_candidate",
        ),
        _check_row(
            "llama_server",
            LLAMA_SERVER_PATH,
            True,
            LLAMA_SERVER_PATH.is_file() and os.access(LLAMA_SERVER_PATH, os.X_OK),
            artifact_field="runner_path",
        ),
    ]
    return checks, {
        "ownership": ownership,
        "capture": capture,
        "model": model,
        "model_path": model_path,
        "gpu_inventory": gpus,
        "idle_gpus": idle,
        "scored_runtime": scored,
    }


class NativeScorer:  # pragma: no cover - exercised by the live entrypoint.
    """Own one llama.cpp model and expose exact first-token option logits."""

    def __init__(self, model_path: Path, gpu_index: int) -> None:
        self.model_path = model_path
        self.gpu_index = gpu_index
        self.llm: Any = None
        self.backend: Any = None
        self.option_token_ids: dict[str, int] = {}

    def load(self) -> None:
        """Load all supported layers on the one device selected by the lease."""

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        import llama_cpp  # noqa: PLC0415

        self.backend = llama_cpp
        self.llm = llama_cpp.Llama(
            model_path=str(self.model_path),
            n_ctx=1024,
            n_batch=512,
            n_gpu_layers=-1,
            split_mode=0,
            main_gpu=0,
            logits_all=True,
            verbose=False,
        )
        for label in ("A", "B"):
            ids = self.tokenize(f" {label}", add_bos=False)
            if len(ids) != 1:
                raise RuntimeError(f"option_label_not_single_token:{label}:{ids}")
            self.option_token_ids[label] = ids[0]

    def tokenize(self, text: str, *, add_bos: bool) -> list[int]:
        """Use the embedded GGUF tokenizer for prompt and option identity."""

        try:
            return list(self.llm.tokenize(text.encode(), add_bos=add_bos, special=True))
        except TypeError:
            return list(self.llm.tokenize(text.encode(), add_bos=add_bos))

    def score(self, prompt_text: str) -> JsonDict:
        """Reset state and retain input IDs plus both raw option logits."""

        input_ids = self.tokenize(prompt_text, add_bos=True)
        self.llm.reset()
        self.llm.eval(input_ids)
        logits = self.llm.scores[len(input_ids) - 1]
        option_logits = {
            label: float(logits[token_id]) for label, token_id in self.option_token_ids.items()
        }
        return {
            "input_ids": input_ids,
            "prompt_token_count": len(input_ids),
            "option_token_ids": deepcopy(self.option_token_ids),
            "option_logits": option_logits,
            "probabilities": option_distribution(option_logits),
            "state_reset": True,
        }

    def close(self) -> None:
        """Release native model memory before the same lease starts the server."""

        if self.llm is not None:
            close = getattr(self.llm, "close", None)
            if callable(close):
                close()
        self.llm = None
        gc.collect()


def _json_request(
    url: str, payload: Mapping[str, Any], timeout: float = 60.0
) -> JsonDict:  # pragma: no cover - live loopback transport.
    """Send one local JSON request without adding a remote dependency."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    return dict(value) if isinstance(value, Mapping) else {}


def _server_option_logits(
    response: Mapping[str, Any], option_token_ids: Mapping[str, int]
) -> dict[str, float]:
    """Extract both returned label log-probabilities or report unavailability."""

    positions = response.get("completion_probabilities")
    first = positions[0] if isinstance(positions, list) and positions else {}
    top = first.get("top_logprobs") if isinstance(first, Mapping) else []
    found: dict[str, float] = {}
    for item in top if isinstance(top, list) else []:
        if not isinstance(item, Mapping):
            continue
        token_id = item.get("id")
        token_text = str(item.get("token") or "")
        for label, expected_id in option_token_ids.items():
            if token_id == expected_id or token_text == f" {label}":
                value = item.get("logprob")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    found[label] = float(value)
    if set(found) != {"A", "B"}:
        raise ValueError(f"missing_option_logits:{sorted(set(('A', 'B')) - set(found))}")
    return found


def _free_port() -> int:  # pragma: no cover - live server allocation.
    """Reserve a loopback port briefly before one owned server starts."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _stream_server_output(
    process: subprocess.Popen[str], log_path: Path
) -> threading.Thread:  # pragma: no cover - live subprocess drain.
    """Drain only the owned child's pipe so it cannot block or appear silent."""

    def drain() -> None:
        with log_path.open("w", encoding="utf-8") as log:
            if process.stdout is None:
                return
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[exp7463-server] {line.rstrip()}", flush=True)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _start_server(
    model_path: Path, gpu_index: int, raw_dir: Path, started: float
) -> tuple[subprocess.Popen[str], threading.Thread, int]:  # pragma: no cover
    """Start one owned llama-server and wait with truthful heartbeats."""

    port = _free_port()
    argv = [
        str(LLAMA_SERVER_PATH),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "1024",
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--offline",
        "--no-webui",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    progress(started, "local_server_load", "before_model_load", argv_hash=canonical_hash(argv))
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=environment,
    )
    thread = _stream_server_output(process, raw_dir / "llama-server.log")
    deadline = time.monotonic() + 240
    last_heartbeat = time.monotonic()
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"owned_llama_server_exit:{process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if 200 <= response.status < 300:
                    progress(started, "local_server_load", "after_model_load", pid=process.pid)
                    return process, thread, port
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() - last_heartbeat >= 60:
            progress(started, "local_server_load", "pending", pid=process.pid)
            last_heartbeat = time.monotonic()
        time.sleep(1)
    raise TimeoutError("owned_llama_server_load_timeout")


def _stop_owned_server(
    process: subprocess.Popen[str], thread: threading.Thread, started: float
) -> int:  # pragma: no cover
    """Stop only the process group created by this experiment."""

    progress(started, "local_server_unload", "before_owned_stop", pid=process.pid)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    thread.join(timeout=5)
    progress(started, "local_server_unload", "after_owned_stop", exit_code=process.returncode)
    return int(process.returncode or 0)


def _gpu_memory(gpu_index: int) -> int:  # pragma: no cover - live CUDA receipt.
    """Read one device's used memory for requested-versus-observed offload."""

    for row in _gpu_inventory():
        if row["index"] == gpu_index:
            return int(row["memory_used_mb"])
    return 0


def _score_server(
    port: int,
    prompt_row: Mapping[str, Any],
    option_token_ids: Mapping[str, int],
) -> JsonDict:  # pragma: no cover
    """Emit one token only because llama-server exposes top log-probs on generation."""

    prompt_text = str(prompt_row["prompt"])
    tokenized = _json_request(
        f"http://127.0.0.1:{port}/tokenize",
        {"content": prompt_text, "add_special": True},
    )
    response = _json_request(
        f"http://127.0.0.1:{port}/completion",
        {
            "prompt": prompt_text,
            "n_predict": 1,
            "temperature": 0.0,
            "n_probs": 128,
            "cache_prompt": False,
        },
    )
    option_logits = _server_option_logits(response, option_token_ids)
    tokens = tokenized.get("tokens")
    input_ids = [int(value) for value in tokens] if isinstance(tokens, list) else []
    return {
        "input_ids": input_ids,
        "prompt_token_count": len(input_ids),
        "option_token_ids": dict(option_token_ids),
        "option_logits": option_logits,
        "probabilities": option_distribution(option_logits),
        "state_reset": True,
        "response_sha256": canonical_hash(response),
        "emitted_text": str(response.get("content") or ""),
    }


def _source_hashes(
    root: Path, model_path: Path
) -> JsonDict:  # pragma: no cover - hashes multi-GB live model bytes.
    """Bind current code, historical receipts, manifest, runner, and model bytes."""

    paths = (
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        SPEC_PATH,
        KERNEL_MANIFEST_PATH,
        OWNERSHIP_RESULT_PATH,
        CAPTURE_RESULT_PATH,
        WORLD_MODEL_PATH,
    )
    result = {
        path.as_posix(): {"path": path.as_posix(), "sha256": sha256_file(root / path)}
        for path in paths
    }
    result["model_gguf"] = {
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "size_bytes": model_path.stat().st_size,
    }
    result["llama_server"] = {
        "path": str(LLAMA_SERVER_PATH),
        "sha256": sha256_file(LLAMA_SERVER_PATH),
        "size_bytes": LLAMA_SERVER_PATH.stat().st_size,
    }
    return result


def _span(
    phase: str, phase_started: float, run_started: float, completed_units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - live orchestration timing.
    """Close one real phase with a monotonic checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
    }


def _terminal_artifact(
    *,
    started_at: str,
    started_ns: int,
    preconditions: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    controls: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    runtime_manifest: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    durations: Mapping[str, float],
) -> JsonDict:  # pragma: no cover - live evidence assembly.
    """Build the measured candidate after inference and independent reduction."""

    local = reduce_parity_rows(rows, expected_units=N_DEVELOPMENT)
    scored_runtime = dict(context["scored_runtime"])
    gates = _base_gates(controls, local, scored_runtime)
    scores = derive_scores(controls, local, {"passed": False})
    counts = _zero_counts()
    for event in events:
        operation = str(event["operation"])
        state = str(event["state"])
        counts[f"{operation}_{state}"] += 1
    ended_ns = time.monotonic_ns()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "complete_scored_runtime_unavailable",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "monotonic_clock": "time.monotonic_ns",
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": MODEL_SPECS,
        "model_invoked": bool(events),
        "invocation_counts": counts,
        "current_invocation_events": [deepcopy(dict(row)) for row in events],
        "inference_substrate": "native_llama_cpp_logits_plus_owned_llama_server_one_token",
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": (ended_ns - started_ns) / 1_000_000_000,
        "duration_breakdown_s": dict(durations),
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": {"ordering": 7463, "audit": 7464, "bootstrap": None, "fit": None},
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [deepcopy(dict(row)) for row in rows],
        "positive_control_rows": [deepcopy(dict(row)) for row in controls],
        "sample_size_budget": {
            "planned_independent_units": 72,
            "attempted_independent_units": len(controls) + len(rows),
            "complete_independent_units": sum(row.get("disposition") == "complete" for row in rows)
            + sum(row.get("disposition") == "complete" for row in controls),
            "failed_independent_units": sum(row.get("disposition") == "failed" for row in rows)
            + sum(row.get("disposition") == "failed" for row in controls),
            "censored_independent_units": sum(row.get("disposition") == "censored" for row in rows),
            "unstarted_independent_units": N_DEVELOPMENT,
            "server_requests_planned": N_DEVELOPMENT,
            "server_requests_attempted": counts["generation_calls_attempted"],
            "server_request_cap": MAX_SERVER_REQUESTS,
            "live_work_cap_s": MAX_LIVE_SECONDS,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "honest_verdict": (
            "complete_scored_runtime_unavailable_local_parity_passed"
            if local["passed"]
            else "complete_scored_runtime_unavailable_local_parity_null"
        ),
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": {},
        **scores,
        "local_runtime_reduction": local,
        "scored_runtime_reduction": {
            "passed": False,
            "status": "blocked_no_run",
            "failed_check": scored_runtime.get("failed_check"),
        },
        "runtime_manifest": deepcopy(dict(runtime_manifest)),
        "proposed_request_change": deepcopy(scored_runtime["proposed_request_change"]),
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "pending",
            "numbered_e2e": "not_applicable_isolated_readout_experiment",
        },
        "deployment_authorized": False,
        "scored_path_changed": False,
        "production_defaults_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _event(
    call_id: str, operation: str, state: str
) -> JsonDict:  # pragma: no cover - live ledger boundary.
    """Create one current-work event at the boundary where it occurs."""

    return {
        "call_id": call_id,
        "operation": operation,
        "state": state,
        "monotonic_ns": time.monotonic_ns(),
        "scope": "current",
    }


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - live capability E2E.
    """Run owned native/server measurements, scoped checks, and atomic publish."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR / f"attempt-{started_ns}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []
    events: list[JsonDict] = []
    durations = {
        "model_load": 0.0,
        "forward": 0.0,
        "generation": 0.0,
        "numeric_reduction": 0.0,
        "validation": 0.0,
    }

    phase_started = time.monotonic()
    progress(started, "preconditions", "start", completed_units=0)
    checks, context = collect_preconditions(root)
    spans.append(
        _span("preconditions", phase_started, started, len(checks), "inputs_authenticated")
    )
    progress(
        started,
        "preconditions",
        "end",
        completed_units=len(checks),
        passed=all(row["passed"] for row in checks),
        scored_available=context["scored_runtime"]["available"],
    )
    if not all(row["passed"] for row in checks):
        raise RuntimeError("required_local_precondition_failed:" + str(_gate_summary(checks)))

    model_path = Path(context["model_path"])
    gpu = dict(context["idle_gpus"][0])
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=EXPERIMENT_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["memory_used_mb"]),
        ttl_s=MAX_LIVE_SECONDS + 300,
    )
    owner = lease.owner_receipt()
    lease.transition("admitted")
    lease.transition("loading")
    controls: list[JsonDict] = []
    native_development: dict[str, JsonDict] = {}
    rows: list[JsonDict] = []
    runtime_manifest: JsonDict = {}
    server: subprocess.Popen[str] | None = None
    server_thread: threading.Thread | None = None
    server_exit = 1
    inference_ok = False
    try:
        phase_started = time.monotonic()
        progress(started, "native_load", "before_model_load", completed_units=0)
        load_started = time.monotonic()
        events.append(_event("native-load", "model_loads", "attempted"))
        scorer = NativeScorer(model_path, int(gpu["index"]))
        scorer.load()
        events.append(_event("native-load", "model_loads", "completed"))
        durations["model_load"] += time.monotonic() - load_started
        resident_mb = _gpu_memory(int(gpu["index"]))
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        progress(
            started,
            "native_load",
            "after_model_load",
            completed_units=1,
            observed_vram_mb=resident_mb,
        )
        runtime_manifest["native_llama_cpp"] = {
            "model_id": MODEL_HF_ID,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "model_representation": "GGUF",
            "quantization": "Q4_K_M",
            "tokenizer": "embedded_gguf",
            "prompt_protocol_sha256": DEVELOPMENT_PROMPTS_SHA256,
            "runner_path": str(Path(scorer.backend.__file__).resolve()),
            "runner_sha256": sha256_file(Path(scorer.backend.__file__).resolve()),
            "requested_offload": {"n_gpu_layers": -1, "device_index": gpu["index"]},
            "observed_offload": {
                "device_uuid": gpu["uuid"],
                "resident_vram_mb": resident_mb,
                "cuda_offload": resident_mb - int(gpu["memory_used_mb"]) > 1_000,
            },
            "lease_id": owner["lease_id"],
        }

        phase_started = time.monotonic()
        progress(started, "native_controls", "before_forward_loop", completed_units=0)
        for index, row in enumerate(freeze_positive_controls()):
            call_id = f"native-control-{index:02d}"
            events.append(_event(call_id, "forward_calls", "attempted"))
            call_started = time.monotonic()
            scored = scorer.score(str(row["prompt"]))
            durations["forward"] += time.monotonic() - call_started
            events.append(_event(call_id, "forward_calls", "completed"))
            predicted = max(("A", "B"), key=lambda label: scored["probabilities"][label])
            spread = abs(scored["probabilities"]["A"] - scored["probabilities"]["B"])
            controls.append(
                {
                    **deepcopy(row),
                    "row_kind": "positive_control",
                    "disposition": "complete",
                    "native": scored,
                    "predicted_label": predicted,
                    "correct": predicted == row["expected_label"],
                    "nonuniform": spread > 1e-6,
                    "offload_observed": runtime_manifest["native_llama_cpp"]["observed_offload"][
                        "cuda_offload"
                    ],
                    "lease_id": owner["lease_id"],
                }
            )
            progress(
                started,
                "native_controls",
                "unit_complete",
                completed_units=index + 1,
                correct=controls[-1]["correct"],
            )
        if (
            derive_scores(controls, {"passed": False}, {"passed": False})[
                "native_readout_ready_score"
            ]
            != 1
        ):
            raise RuntimeError("native_positive_control_failed")
        spans.append(
            _span("native_controls", phase_started, started, len(controls), "controls_complete")
        )
        progress(started, "native_controls", "after_forward_loop", completed_units=8)

        phase_started = time.monotonic()
        progress(started, "native_development", "before_forward_loop", completed_units=0)
        for index, row in enumerate(freeze_development_prompts()):
            call_id = f"native-development-{index:02d}"
            events.append(_event(call_id, "forward_calls", "attempted"))
            call_started = time.monotonic()
            native_development[str(row["unit_id"])] = scorer.score(str(row["prompt"]))
            durations["forward"] += time.monotonic() - call_started
            events.append(_event(call_id, "forward_calls", "completed"))
            if (index + 1) % 8 == 0:
                progress(
                    started,
                    "native_development",
                    "checkpoint",
                    completed_units=index + 1,
                )
        spans.append(
            _span(
                "native_development",
                phase_started,
                started,
                len(native_development),
                "native_logits_complete",
            )
        )
        progress(started, "native_development", "after_forward_loop", completed_units=64)

        progress(started, "native_unload", "before_model_unload")
        scorer.close()
        progress(started, "native_unload", "after_model_unload")

        phase_started = time.monotonic()
        load_started = time.monotonic()
        events.append(_event("local-server-load", "model_loads", "attempted"))
        server, server_thread, port = _start_server(model_path, int(gpu["index"]), raw_dir, started)
        events.append(_event("local-server-load", "model_loads", "completed"))
        durations["model_load"] += time.monotonic() - load_started
        server_vram = _gpu_memory(int(gpu["index"]))
        runtime_manifest["local_llama_server"] = {
            "model_id": MODEL_HF_ID,
            "model_path": str(model_path),
            "model_sha256": runtime_manifest["native_llama_cpp"]["model_sha256"],
            "model_representation": "GGUF",
            "quantization": "Q4_K_M",
            "tokenizer": "embedded_gguf",
            "prompt_protocol_sha256": DEVELOPMENT_PROMPTS_SHA256,
            "runner_path": str(LLAMA_SERVER_PATH),
            "runner_sha256": sha256_file(LLAMA_SERVER_PATH),
            "request_schema": {
                "endpoint": "/completion",
                "n_predict": 1,
                "n_probs": 128,
                "temperature": 0.0,
                "logit_bias": None,
            },
            "requested_offload": {"n_gpu_layers": "all", "device_index": gpu["index"]},
            "observed_offload": {
                "device_uuid": gpu["uuid"],
                "resident_vram_mb": server_vram,
                "cuda_offload": server_vram - int(gpu["memory_used_mb"]) > 1_000,
            },
            "lease_id": owner["lease_id"],
            "owned_pid": server.pid,
        }
        spans.append(_span("local_server_load", phase_started, started, 1, "server_healthy"))

        phase_started = time.monotonic()
        progress(started, "local_parity", "before_generation_loop", completed_units=0)
        for index, prompt_row in enumerate(freeze_development_prompts()):
            if time.monotonic() - started >= MAX_LIVE_SECONDS:
                raise TimeoutError("live_work_cap_reached")
            call_id = f"local-server-{index:02d}"
            events.append(_event(call_id, "generation_calls", "attempted"))
            call_started = time.monotonic()
            native = native_development[str(prompt_row["unit_id"])]
            try:
                local = _score_server(port, prompt_row, native["option_token_ids"])
                disposition = "complete"
                error = None
                events.append(_event(call_id, "generation_calls", "completed"))
            except (OSError, ValueError, urllib.error.URLError) as exc:
                local = {"error": f"{type(exc).__name__}:{exc}"}
                disposition = "unavailable"
                error = str(exc)
                events.append(_event(call_id, "generation_calls", "failed"))
            durations["generation"] += time.monotonic() - call_started
            rows.append(
                {
                    **deepcopy(prompt_row),
                    "row_kind": "parity_pair",
                    "disposition": disposition,
                    "error": error,
                    "native": native,
                    "local_server": local,
                    "scored_runtime": {
                        "disposition": "unstarted",
                        "reason": context["scored_runtime"]["failed_check"],
                    },
                    "input_ids_equal": local.get("input_ids") == native["input_ids"],
                    "weights_equal": True,
                    "quantization_equal": True,
                    "tokenizer_equal": True,
                    "logit_bias": None,
                    "emitted_tokens": 1,
                    "local_parity_score": (
                        1.0
                        - 0.5
                        * sum(
                            abs(
                                float(native["probabilities"][label])
                                - float(local["probabilities"][label])
                            )
                            for label in ("A", "B")
                        )
                        if disposition == "complete"
                        else None
                    ),
                    "lease_id": owner["lease_id"],
                }
            )
            if (index + 1) % 8 == 0:
                progress(
                    started,
                    "local_parity",
                    "checkpoint",
                    completed_units=index + 1,
                    unavailable=sum(row["disposition"] != "complete" for row in rows),
                )
        spans.append(_span("local_parity", phase_started, started, len(rows), "raw_pairs_complete"))
        progress(started, "local_parity", "after_generation_loop", completed_units=64)
        inference_ok = True
    finally:
        if server is not None and server_thread is not None:
            server_exit = _stop_owned_server(server, server_thread, started)
        lease.transition("unloading")
        after_mb = _gpu_memory(int(gpu["index"]))
        lease.transition(
            "validating",
            vram_mb=after_mb,
            exit_code=0 if inference_ok and server_exit in (0, -signal.SIGTERM) else 1,
            unload_observed=after_mb <= int(gpu["memory_used_mb"]) + 1_024,
        )
        lease.transition("terminal_complete" if inference_ok else "terminal_blocked")
        release = lease.release()

    runtime_manifest["lease"] = {**owner, "release": release}
    runtime_manifest["scored_vllm"] = {
        **deepcopy(context["scored_runtime"]),
        "model_id": MODEL_HF_ID,
        "model_representation": "NVFP4_safetensors_declared_not_mounted_here",
        "tokenizer": "attached_runtime_unverified",
        "device": "NvidiaRtxPro6000_declared_not_available_on_host",
        "comparison_class": "blocked_no_run",
    }

    phase_started = time.monotonic()
    progress(started, "hashes", "before_model_and_source_hashes", completed_units=0)
    source_hashes = _source_hashes(root, model_path)
    spans.append(
        _span("source_hashes", phase_started, started, len(source_hashes), "hashes_complete")
    )
    progress(started, "hashes", "after_model_and_source_hashes", completed_units=len(source_hashes))

    reduction_started = time.monotonic()
    candidate = _terminal_artifact(
        started_at=started_at,
        started_ns=started_ns,
        preconditions=checks,
        context=context,
        controls=controls,
        rows=rows,
        events=events,
        runtime_manifest=runtime_manifest,
        source_hashes=source_hashes,
        spans=spans,
        durations=durations,
    )
    durations["numeric_reduction"] = time.monotonic() - reduction_started
    candidate["duration_breakdown_s"] = dict(durations)
    candidate["reproducibility_checksum"] = artifact_checksum(candidate)
    candidate_path = raw_dir / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    validation_started = time.monotonic()
    private = Path(tempfile.mkdtemp(prefix="exp7463-validation-", dir="/tmp"))
    progress(started, "affected_validation", "before_subprocesses", completed_units=0)
    affected_commands = build_validation_commands(root, private)
    affected_receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in affected_commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60,
    )
    affected = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected_receipts)
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected_receipts),
        passed=affected.get("passed"),
    )

    preterminal = deepcopy(candidate)
    preterminal["validation_receipts"] = [deepcopy(dict(row)) for row in affected_receipts]
    preterminal["field_principles"] = _field_principles(tuple(preterminal))
    preterminal["reproducibility_checksum"] = artifact_checksum(preterminal)
    atomic_json(candidate_path, preterminal)
    terminal_commands = _terminal_commands(root, candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                command,
                "safety" if command.name == "adversarial_verify" else "completion",
                True,
            )
            for command in terminal_commands
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60,
    )
    durations["validation"] = time.monotonic() - validation_started
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
    )
    final = finalize_validation(candidate, [*affected_receipts, *terminal_receipts])
    final["duration_breakdown_s"] = dict(durations)
    final["completed_at_utc"] = utc_now()
    final["ended_monotonic_ns"] = time.monotonic_ns()
    final["duration_s"] = (final["ended_monotonic_ns"] - started_ns) / 1_000_000_000
    final["reproducibility_checksum"] = artifact_checksum(final)
    errors = validate_artifact(final, require_validation=True)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    progress(started, "publish", "before_atomic_publish", path=output)
    atomic_json(output, final)
    progress(
        started,
        "publish",
        "after_atomic_publish",
        completed_units=len(rows) + len(controls),
        verdict=final["honest_verdict"],
    )
    return final


def date_argument(value: str) -> str:
    """Accept only the fixed execution date used by the V654 protocol."""

    if value != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{value}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the live experiment or cold-validate one measured candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        value = _load_json(args.validate)
        errors = validate_artifact(value, require_validation=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    result = run_experiment(
        root=REPO_ROOT,
        run_date=args.date,
        output_path=args.output.resolve() if args.output else None,
    )
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "honest_verdict": result["honest_verdict"],
                "native_readout_ready_score": result["native_readout_ready_score"],
                "local_runtime_parity_score": result["local_runtime_parity_score"],
                "scored_runtime_parity_score": result["scored_runtime_parity_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
