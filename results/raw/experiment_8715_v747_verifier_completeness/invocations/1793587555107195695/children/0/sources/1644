#!/usr/bin/env python3
"""Exp 5324: stabilize a native GGUF backend candidate with repeated receipts.

Spec refs: REQ-VERIFY-5324, SCENARIO-VERIFY-5324.

This module is a runtime stability check only. It replays the exact native
llama.cpp command selected by Exp 5323 for one mandated local SOTA GGUF model
and records repeated bounded receipts. It deliberately makes no quality,
accuracy, verifier, solver, benchmark, or memory-usefulness claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5323_native_gguf_backend_flag_bisect_v486 as exp5323
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
PreconditionsProvider = Callable[[], JsonDict]
RuntimeProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5324_runtime_receipt_stabilization_v486"
MILESTONE = "2026.07.486"
RESULT_RELATIVE_PATH = Path("results/experiment_5324_runtime_receipt_stabilization_v486.json")
SCHEMA = "carnot.experiment_5324.runtime_receipt_stabilization.v486"
INFERENCE_SUBSTRATE = "local_native_llama_cpp_stability_receipts"
SPEC_REFS = ("REQ-VERIFY-5324", "SCENARIO-VERIFY-5324")

MANDATED_MODEL_SPECS = exp5323.MANDATED_MODEL_SPECS
PROMPT = exp5323.PROMPT
RANDOM_SEED = 5324
MIN_REPEATS = 3
VALID_FAILURE_CLASSES = (
    "command_drift",
    "memory_pressure",
    "model_specific_assertion",
    "timeout",
    "missing_binary",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5324 local native GGUF runtime receipt stabilization.",
    "milestone": "Milestone accountability for the V486 stability decision.",
    "status": "Machine-readable terminal state for downstream runtime gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether the "
        "native SOTA runtime receipt is repeatably stable."
    ),
    "inference_substrate": (
        "Declares local_native_llama_cpp_stability_receipts so the artifact is read as "
        "repeated native runtime receipts, not a quality benchmark."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF repository IDs and concrete local GGUF "
        "cache status without AutoTokenizer fallback."
    ),
    "preconditions_checked": (
        "Records GPU visibility, llama.cpp binary version, selected model file "
        "presence, VRAM, CUDA evidence, and the exact Exp5323 command before replay."
    ),
    "selected_model_spec": (
        "Binds the repeated receipts to the one mandated model selected by the Exp5323 "
        "successful backend command."
    ),
    "selected_backend_command": (
        "Preserves the exact Exp5323 backend command so stability replay cannot "
        "silently drift to easier flags or a different model."
    ),
    "repeated_receipts": (
        "Records at least three bounded replay receipts with timing, 8-token "
        "completion, GPU memory, offload, timeout, and stderr evidence."
    ),
    "stability_failure_class": (
        "Names the dominant stability blocker for downstream gates without inflating "
        "partial receipt success into an unblock."
    ),
    "tests_run": (
        "Commands run to validate the stability module, artifact schema, new-code "
        "coverage, and required repository test status."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_model_spec",
    "selected_backend_command",
    "repeated_receipts",
    "stability_failure_class",
    "sota_runtime_unblocked_stable",
    "quality_claim_permitted",
    "tests_run",
)
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_model_spec",
    "selected_backend_command",
    "repeated_receipts",
    "stability_failure_class",
    "tests_run",
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _prior_candidate(prior_artifact: Mapping[str, Any]) -> JsonDict | None:
    if _raw_or_wrapped_value(prior_artifact, "sota_backend_candidate_ready") is not True:
        return None
    if _raw_or_wrapped_value(prior_artifact, "runtime_unblocked_min_one_mandated") is not True:
        return None
    candidate = _raw_or_wrapped_value(prior_artifact, "best_backend_command")
    if not isinstance(candidate, Mapping):
        return None
    command = candidate.get("command")
    if not isinstance(command, list) or not command or not candidate.get("model_role"):
        return None
    return dict(candidate)


def _resolve_model_specs(model_resolver: ModelResolver) -> JsonDict:
    return {
        str(spec["role"]): exp5323._resolve_model_spec(spec, model_resolver)
        for spec in MANDATED_MODEL_SPECS
    }


def _selected_model_spec(candidate: Mapping[str, Any] | None, model_specs: Mapping[str, Any]) -> JsonDict | None:
    if candidate is None:
        return None
    role = str(candidate.get("model_role") or "")
    selected = model_specs.get(role)
    return dict(selected) if isinstance(selected, Mapping) else None


def _candidate_command(candidate: Mapping[str, Any] | None) -> list[str] | None:
    if candidate is None:
        return None
    command = candidate.get("command")
    return list(command) if isinstance(command, list) and command else None


def _build_replay_variant(candidate: Mapping[str, Any]) -> JsonDict:
    command = list(candidate["command"])
    return {
        "name": str(candidate.get("backend_variant") or "exp5323-selected-command"),
        "backend_kind": str(candidate.get("backend_kind") or Path(command[0]).name),
        "command": command,
        "model_path": str(candidate.get("model_path") or ""),
        "context": int(candidate.get("context") or exp5323.DEFAULT_CONTEXT),
        "batch": int(candidate.get("batch") or exp5323.DEFAULT_BATCH),
        "ubatch": int(candidate.get("ubatch") or exp5323.DEFAULT_UBATCH),
        "gpu_layers": str(candidate.get("gpu_layers") or exp5323.DEFAULT_GPU_LAYERS),
        "split_mode": exp5323.DEFAULT_SPLIT_MODE,
        "tensor_split": candidate.get("tensor_split"),
        "prompt": str(candidate.get("prompt") or PROMPT),
        "n_predict": int(candidate.get("n_predict") or exp5323.N_PREDICT),
        "timeout_s": float(candidate.get("timeout_s") or exp5323.DEFAULT_TIMEOUT_S),
    }


def _precondition_blockers(
    *,
    preconditions: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    selected_model: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    command = _candidate_command(candidate)
    if candidate is None or command is None:
        blockers.append("exp5323_candidate_unavailable")
    if not preconditions.get("gpu_visible"):
        blockers.append("gpu_not_visible")
    if int(preconditions.get("free_vram_mb") or 0) <= 0:
        blockers.append("free_vram_unavailable")
    if candidate is not None:
        prior_delta = int(candidate.get("gpu_memory_delta_mb") or 0)
        if prior_delta > 0 and int(preconditions.get("free_vram_mb") or 0) < prior_delta:
            blockers.append("free_vram_below_exp5323_delta")
    if command is not None:
        if not Path(str(command[0])).is_file():
            blockers.append("selected_binary_missing")
    if selected_model is None or selected_model.get("status") != "local_gguf_resolved":
        blockers.append("selected_model_file_missing")
    else:
        selected_path = str(selected_model.get("model_path") or "")
        candidate_path = str((candidate or {}).get("model_path") or "")
        if not selected_path or not Path(selected_path).is_file():
            blockers.append("selected_model_file_missing")
        elif candidate_path and selected_path != candidate_path:
            blockers.append("selected_command_model_path_drift")
    if candidate is not None and not preconditions.get("cuda_backend_evidence"):
        blockers.append("native_llama_cpp_cuda_evidence_missing")
    return list(dict.fromkeys(blockers))


def classify_precondition_failure(blockers: Sequence[str]) -> str:
    if any("candidate_unavailable" in blocker or "drift" in blocker for blocker in blockers):
        return "command_drift"
    if any("binary_missing" in blocker or "model_file_missing" in blocker for blocker in blockers):
        return "missing_binary"
    if any("vram" in blocker or "memory" in blocker for blocker in blockers):
        return "memory_pressure"
    return "command_drift"


def _stderr_summary(text: str) -> str:
    return "\n".join(str(text).strip().splitlines()[-12:])[-1200:]


def _receipt_ready(receipt: Mapping[str, Any]) -> bool:
    return bool(
        exp5323._attempt_is_ready(receipt)
        and receipt.get("timeout_class") == "completed_no_timeout"
        and not receipt.get("timed_out")
    )


def classify_stability_failure(
    receipts: Sequence[Mapping[str, Any]], blockers: Sequence[str]
) -> str:
    if blockers:
        return classify_precondition_failure(blockers)
    text = "\n".join(str(row.get("stderr_summary", "")) for row in receipts).lower()
    classes = [str(row.get("timeout_class", "")) for row in receipts]
    if "out of memory" in text or "cuda error" in text or "cublas" in text:
        return "memory_pressure"
    if any("assert" in cls for cls in classes) or "ggml_assert" in text or "n_tokens_all" in text:
        return "model_specific_assertion"
    if any(cls.startswith("timeout") for cls in classes):
        return "timeout"
    return "command_drift"


def default_runtime_probe(
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float,
    run_index: int,
) -> JsonDict:  # pragma: no cover - delegates to the live Exp5323 subprocess probe
    _ = run_index
    return exp5323.default_runtime_probe(
        model_spec=model_spec,
        variant=variant,
        timeout_s=timeout_s,
    )


def _normalise_replay_receipt(
    receipt: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float,
    run_index: int,
) -> JsonDict:
    attempt = exp5323._normalise_attempt(
        receipt,
        model_spec=model_spec,
        variant=variant,
        timeout_s=timeout_s,
    )
    attempt["run_index"] = run_index
    attempt["stderr_summary"] = _stderr_summary(attempt.get("stderr_tail", ""))
    return attempt


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    prior_artifact_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    preconditions_provider: PreconditionsProvider | None = None,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    tests_run: Sequence[Any] | None = None,
    repeats: int = MIN_REPEATS,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    prior_artifact_path = prior_artifact_path or root / exp5323.RESULT_RELATIVE_PATH
    prior_artifact = _read_json(prior_artifact_path)
    candidate = _prior_candidate(prior_artifact)
    preconditions_provider = preconditions_provider or (lambda: exp5323.collect_preconditions(root))
    preconditions = dict(preconditions_provider())
    model_specs = _resolve_model_specs(model_resolver)
    selected_model = _selected_model_spec(candidate, model_specs)
    command = _candidate_command(candidate)
    preconditions["exp5323_artifact_path"] = str(prior_artifact_path)
    preconditions["exp5323_candidate_ready"] = candidate is not None
    preconditions["selected_backend_command"] = command
    preconditions["selected_model_role"] = (candidate or {}).get("model_role") if candidate else None
    preconditions["selected_model_file_present"] = bool(
        selected_model and selected_model.get("model_path") and Path(str(selected_model["model_path"])).is_file()
    )
    preconditions["autotokenizer_used"] = False
    blockers = _precondition_blockers(
        preconditions=preconditions,
        candidate=candidate,
        selected_model=selected_model,
    )
    preconditions["blocked_preconditions"] = blockers

    receipts: list[JsonDict] = []
    selected_command = dict(candidate) if candidate is not None else None
    if not blockers and candidate is not None and selected_model is not None:
        variant = _build_replay_variant(candidate)
        timeout_s = float(variant["timeout_s"])
        for run_index in range(1, max(MIN_REPEATS, repeats) + 1):
            raw_receipt = runtime_probe(
                model_spec=selected_model,
                variant=variant,
                timeout_s=timeout_s,
                run_index=run_index,
            )
            receipts.append(
                _normalise_replay_receipt(
                    raw_receipt,
                    model_spec=selected_model,
                    variant=variant,
                    timeout_s=timeout_s,
                    run_index=run_index,
                )
            )

    stable = len(receipts) >= MIN_REPEATS and all(_receipt_ready(receipt) for receipt in receipts)
    failure_class = "none" if stable else classify_stability_failure(receipts, blockers)
    status = "complete" if stable else "blocked"
    if stable:
        honest = (
            "complete: local_native_llama_cpp_stability_receipts="
            f"{selected_command['model_role']}:{selected_command['backend_kind']}"
        )
    else:
        honest = f"blocked_sota_runtime_unblocked_stable_false: {failure_class}"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "preconditions_checked": _wrap("preconditions_checked", preconditions),
        "selected_model_spec": _wrap("selected_model_spec", selected_model),
        "selected_backend_command": _wrap("selected_backend_command", selected_command),
        "repeated_receipts": _wrap("repeated_receipts", receipts),
        "stability_failure_class": _wrap("stability_failure_class", failure_class),
        "sota_runtime_unblocked_stable": stable,
        "quality_claim_permitted": False,
        "tests_run": _wrap("tests_run", list(tests_run or [])),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = exp5323.sha16(
        exp5323._stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "prior_candidate": selected_command,
                "selected_model": selected_model,
                "receipts": receipts,
                "stable": stable,
                "failure_class": failure_class,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    if write:
        exp5323.write_json(artifact_path, artifact)
    return artifact


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if not isinstance(value, Mapping):
        return MISSING_WRAPPED_VALUE
    if value.get("principle") != FIELD_PRINCIPLES.get(field):
        return MISSING_WRAPPED_VALUE
    return value.get("value")


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in WRAPPED_FIELDS:
        if field in artifact and _wrapped_value(artifact, field) is MISSING_WRAPPED_VALUE:
            errors.append(f"{field} must be principle-wrapped")
    if _wrapped_value(artifact, "experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if _wrapped_value(artifact, "milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if _wrapped_value(artifact, "status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    honest = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not isinstance(artifact.get("sota_runtime_unblocked_stable"), bool):
        errors.append("sota_runtime_unblocked_stable must be a bare boolean")
    if artifact.get("quality_claim_permitted") is not False:
        errors.append("quality_claim_permitted must be bare false")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        expected_roles = {str(spec["role"]) for spec in MANDATED_MODEL_SPECS}
        if set(model_specs) != expected_roles:
            errors.append("MODEL_SPECS roles mismatch")
        expected_hf = {str(spec["role"]): str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
        for role in expected_roles & set(model_specs):
            spec = model_specs[role]
            if spec.get("hf_id") != expected_hf[role]:
                errors.append("hf_id mismatch for mandated model role")
            if spec.get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")

    selected_command = _wrapped_value(artifact, "selected_backend_command")
    selected_model = _wrapped_value(artifact, "selected_model_spec")
    receipts = _wrapped_value(artifact, "repeated_receipts")
    failure_class = _wrapped_value(artifact, "stability_failure_class")
    tests_run = _wrapped_value(artifact, "tests_run")
    stable = artifact.get("sota_runtime_unblocked_stable")
    if tests_run is not MISSING_WRAPPED_VALUE and not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    if receipts is not MISSING_WRAPPED_VALUE and not isinstance(receipts, list):
        errors.append("repeated_receipts must be a list")
    if stable:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("stable artifact must have complete status")
        if failure_class != "none":
            errors.append("stable artifact must have failure class none")
        if not isinstance(selected_command, Mapping):
            errors.append("selected_backend_command must be an object")
        if not isinstance(selected_model, Mapping):
            errors.append("selected_model_spec must be an object")
        if not isinstance(receipts, list) or len(receipts) < MIN_REPEATS:
            errors.append("stable artifact must contain at least three receipts")
        elif not all(_receipt_ready(receipt) for receipt in receipts):
            errors.append("stable artifact receipts must all be ready")
    else:
        if failure_class not in VALID_FAILURE_CLASSES:
            errors.append("blocked artifact must name failure class")
    if stable is False and isinstance(selected_command, Mapping) and not selected_command.get("command"):
        errors.append("selected_backend_command must preserve command when present")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--prior", type=Path, default=REPO_ROOT / exp5323.RESULT_RELATIVE_PATH)
    parser.add_argument("--repeats", type=int, default=MIN_REPEATS)
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        prior_artifact_path=args.prior,
        repeats=args.repeats,
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5324] status={artifact['status']['value']} "
        f"stable={artifact['sota_runtime_unblocked_stable']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
