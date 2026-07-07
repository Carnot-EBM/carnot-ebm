#!/usr/bin/env python3
"""Exp 5331: stable local GGUF internal-energy receipt harness.

Spec refs: REQ-VERIFY-5331, SCENARIO-VERIFY-5331.

This module is a receipt gate, not a hallucination detector. It checks whether
the stable local GGUF runtime selected by Exp 5324 can emit non-text internal
metadata such as token probabilities, logits, attention data, or hidden-state
proxies for future internal-energy diagnostics. The generated text is recorded
only as a checksum/raw-output receipt and is never scored for quality.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import contextlib
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from carnot import experiment_5323_native_gguf_backend_flag_bisect_v486 as exp5323
from carnot import experiment_5324_runtime_receipt_stabilization_v486 as exp5324


JsonDict = dict[str, Any]
PreconditionsProvider = Callable[[], JsonDict]
OptionSurfaceProvider = Callable[[Mapping[str, Any]], JsonDict]
SignalProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5331_internal_energy_receipt_harness_v486"
MILESTONE = "2026.07.486"
RESULT_RELATIVE_PATH = Path("results/experiment_5331_internal_energy_receipt_harness_v486.json")
RECEIPT_SCHEMA_RELATIVE_PATH = Path(
    "results/experiment_5331_internal_energy_receipt_schema_v486.json"
)
TINY_RECEIPT_RELATIVE_PATH = Path("results/experiment_5331_internal_energy_tiny_receipt_v486.json")
SCHEMA = "carnot.experiment_5331.internal_energy_receipt_harness.v486"
RECEIPT_SCHEMA = "carnot.experiment_5331.internal_energy_receipt_schema.v486"
TINY_RECEIPT_SCHEMA = "carnot.experiment_5331.internal_energy_tiny_receipt.v486"
INFERENCE_SUBSTRATE = "local_sota_internal_signal_receipt"
SPEC_REFS = ("REQ-VERIFY-5331", "SCENARIO-VERIFY-5331")
MODEL_SPECS = exp5323.MANDATED_MODEL_SPECS
PROMPT = "Return exactly OK."
RANDOM_SEED = 5331
N_PREDICT = 2
DEFAULT_TIMEOUT_S = 180.0
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5331 stable local internal-energy receipt harness.",
    "milestone": "Milestone accountability for the V486 internal receipt decision.",
    "status": "Machine-readable terminal state for downstream internal-energy gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether a "
        "reproducible local non-text-score internal receipt exists."
    ),
    "inference_substrate": (
        "Declares local_sota_internal_signal_receipt so the artifact is read as a "
        "stable local GGUF internal-receipt gate, not a generated-text quality scorer."
    ),
    "MODEL_SPECS": (
        "Records the three mandated GGUF model IDs so the receipt harness cannot "
        "silently substitute a legacy, tiny, API, or non-GGUF model."
    ),
    "preconditions_checked": (
        "Records Exp5324 stability, selected backend/model binding, GPU visibility, "
        "model cache status, and backend metadata/API option checks before any internal "
        "receipt claim."
    ),
    "selected_model_spec": (
        "Binds any internal receipt to the stable mandated model selected by Exp5324."
    ),
    "receipt_schema_path": (
        "Points to the tiny internal-receipt schema description, or to a blocked-schema "
        "description naming the missing backend feature precisely."
    ),
    "tests_run": (
        "Commands run to validate the Exp5331 module, artifact schema, new-code "
        "coverage, and required repository test status."
    ),
    "backend_option_surface": (
        "Records the native backend help/API option surface used to decide whether "
        "logits, token probabilities, timing, attention, or hidden proxies are exposed."
    ),
    "raw_output_receipt": (
        "Records prompt/output checksums and timing without interpreting generated text "
        "as answer quality."
    ),
    "missing_backend_features": (
        "Names unavailable backend features precisely so downstream internal-energy work "
        "does not substitute text scoring."
    ),
    "tiny_receipt_path": (
        "Points to the saved tiny non-text internal receipt when one exists; null when "
        "the runtime exposes only text or aggregate timing."
    ),
}

REQUIRED_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_model_spec",
    "receipt_schema_path",
    "tests_run",
)
WRAPPED_FIELDS = REQUIRED_WRAPPED_FIELDS + (
    "backend_option_surface",
    "raw_output_receipt",
    "missing_backend_features",
    "tiny_receipt_path",
)
REQUIRED_ARTIFACT_FIELDS = WRAPPED_FIELDS + (
    "logits_available",
    "token_probability_available",
    "attention_available",
    "hidden_state_proxy_available",
    "token_timing_available",
    "raw_output_receipt_available",
    "external_text_scorer_reopened",
    "internal_signal_receipt_ready",
    "no_quality_claim",
)

CORE_SIGNAL_FIELDS = (
    "logits_available",
    "token_probability_available",
    "attention_available",
    "hidden_state_proxy_available",
)

OPTION_TERMS_RE = re.compile(
    r"logprob|logit|probab|token|timing|perf|attention|hidden|embedding|slots|props|"
    r"metrics|completion",
    re.IGNORECASE,
)
TOKEN_PROBABILITY_PATTERNS = (
    "n_probs",
    "top_logprobs",
    "token_logprobs",
    "completion_probabilities",
    "--logprobs",
)
LOGIT_EXPORT_PATTERNS = ("--logits", "logits_all", "eval_logits", "top_logits")
ATTENTION_EXPORT_PATTERNS = ("dump-attention", "attention_weights", "return_attention")
HIDDEN_PROXY_PATTERNS = ("--embedding", "--embeddings", "/embedding", "embedding")


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def sha16(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _model_specs_from_prior(prior_artifact: Mapping[str, Any]) -> JsonDict:
    prior_specs = _raw_or_wrapped_value(prior_artifact, "MODEL_SPECS")
    prior_specs = prior_specs if isinstance(prior_specs, Mapping) else {}
    out: JsonDict = {}
    for spec in MODEL_SPECS:
        role = str(spec["role"])
        prior = prior_specs.get(role) if isinstance(prior_specs.get(role), Mapping) else {}
        out[role] = {
            "role": role,
            "hf_id": str(spec["hf_id"]),
            "quantization": str(prior.get("quantization") or spec.get("quantization", "Q4_K_M")),
            "model_path": prior.get("model_path"),
            "status": str(prior.get("status") or "missing_local_gguf"),
            "autotokenizer_used": False,
            "file_receipts": prior.get("file_receipts"),
            "metadata": prior.get("metadata"),
        }
    return out


def _selected_model_from_prior(prior_artifact: Mapping[str, Any]) -> JsonDict | None:
    selected = _raw_or_wrapped_value(prior_artifact, "selected_model_spec")
    if not isinstance(selected, Mapping):
        return None
    role = str(selected.get("role") or "")
    expected = {str(spec["role"]): str(spec["hf_id"]) for spec in MODEL_SPECS}
    if role not in expected or selected.get("hf_id") != expected[role]:
        return None
    out = dict(selected)
    out["autotokenizer_used"] = False
    return out


def _selected_command_from_prior(prior_artifact: Mapping[str, Any]) -> JsonDict | None:
    selected = _raw_or_wrapped_value(prior_artifact, "selected_backend_command")
    if not isinstance(selected, Mapping):
        return None
    command = selected.get("command")
    if not isinstance(command, list) or not command:
        return None
    return dict(selected)


def _prior_is_stable(prior_artifact: Mapping[str, Any]) -> bool:
    return bool(_raw_or_wrapped_value(prior_artifact, "sota_runtime_unblocked_stable") is True)


def _selected_model_file_present(selected_model: Mapping[str, Any] | None) -> bool:
    if not selected_model:
        return False
    path = selected_model.get("model_path")
    return bool(path and Path(str(path)).is_file())


def collect_current_preconditions(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover
    return exp5323.collect_preconditions(root)


def _run_command(command: Sequence[str], timeout_s: float = 20.0) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def default_backend_option_surface(
    preconditions: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    receipts: JsonDict = {}
    binary_paths = preconditions.get("binary_paths") if isinstance(preconditions, Mapping) else {}
    for backend in ("llama-cli", "llama-server"):
        path = (binary_paths or {}).get(backend)
        if path and Path(str(path)).is_file():
            receipts[backend] = _run_command([str(path), "--help"], timeout_s=20.0)
        else:
            receipts[backend] = {
                "ok": False,
                "stdout": "",
                "stderr": f"{backend}_binary_missing",
                "command": [str(path or backend), "--help"],
                "returncode": None,
            }
    return summarize_backend_options(receipts)


def _relevant_option_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if OPTION_TERMS_RE.search(line)][:80]


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def summarize_backend_options(help_receipts: Mapping[str, Any]) -> JsonDict:
    summary: JsonDict = {"backends": {}}
    all_text_parts: list[str] = []
    for backend, receipt in help_receipts.items():
        receipt_map = receipt if isinstance(receipt, Mapping) else {}
        text = f"{receipt_map.get('stdout', '')}\n{receipt_map.get('stderr', '')}"
        all_text_parts.append(text)
        summary["backends"][str(backend)] = {
            "ok": bool(receipt_map.get("ok")),
            "help_checksum": sha16(text),
            "relevant_option_lines": _relevant_option_lines(text),
            "command": list(receipt_map.get("command") or []),
        }
    all_text = "\n".join(all_text_parts)
    summary["option_flags"] = {
        "logit_export_option": _contains_any(all_text, LOGIT_EXPORT_PATTERNS),
        "token_probability_option": _contains_any(all_text, TOKEN_PROBABILITY_PATTERNS),
        "attention_export_option": _contains_any(all_text, ATTENTION_EXPORT_PATTERNS),
        "hidden_state_proxy_option": _contains_any(all_text, HIDDEN_PROXY_PATTERNS),
        "aggregate_timing_option": _contains_any(
            all_text, ("--perf", "--show-timings", "timings", "metrics")
        ),
        "raw_output_option": "completion" in all_text.lower() or "prompt" in all_text.lower(),
    }
    return summary


def _normalise_top_logprobs(rows: Any) -> list[JsonDict]:
    normal: list[JsonDict] = []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return normal
    for row in rows[:8]:
        if not isinstance(row, Mapping):
            continue
        normal.append(
            {
                "id": row.get("id"),
                "token_checksum": sha16(str(row.get("token", ""))),
                "logprob": row.get("logprob"),
            }
        )
    return normal


def _normalise_completion_probabilities(value: Any) -> list[JsonDict]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    rows: list[JsonDict] = []
    for row in value[:8]:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "id": row.get("id"),
                "token_checksum": sha16(str(row.get("token", ""))),
                "logprob": row.get("logprob"),
                "top_logprobs": _normalise_top_logprobs(row.get("top_logprobs")),
            }
        )
    return rows


def _surface_available(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("availability") == "available":
            return True
        return any(bool(value.get(key)) for key in ("top_logits", "heads", "embedding"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value)
    return False


def normalise_signal_receipt(
    raw_receipt: Mapping[str, Any], *, prompt: str, backend_kind: str
) -> JsonDict:
    response = raw_receipt.get("response_json")
    response = response if isinstance(response, Mapping) else {}
    content = str(response.get("content") or raw_receipt.get("content") or "")
    completion_probabilities = _normalise_completion_probabilities(
        response.get("completion_probabilities") or raw_receipt.get("completion_probabilities")
    )
    timings = response.get("timings") or raw_receipt.get("timings") or {}
    timings = timings if isinstance(timings, Mapping) else {}
    logits = raw_receipt.get("logits") if isinstance(raw_receipt.get("logits"), Mapping) else {}
    attention = (
        raw_receipt.get("attention") if isinstance(raw_receipt.get("attention"), Mapping) else {}
    )
    hidden_proxy = raw_receipt.get("hidden_state_proxy")
    hidden_proxy = hidden_proxy if isinstance(hidden_proxy, Mapping) else {}
    token_timing_available = bool(
        timings.get("predicted_per_token_ms")
        or timings.get("prompt_per_token_ms")
        or raw_receipt.get("per_token_timings")
    )
    return {
        "status": str(raw_receipt.get("status") or "unknown"),
        "backend_kind": str(raw_receipt.get("backend_kind") or backend_kind),
        "endpoint": raw_receipt.get("endpoint"),
        "prompt_checksum": sha16(prompt),
        "wall_clock_s": raw_receipt.get("wall_clock_s"),
        "raw_output": {
            "availability": "available" if content else "capability_absent",
            "content_checksum": sha16(content) if content else None,
            "content_preview": content[:80],
            "tokens_predicted": response.get("tokens_predicted"),
            "tokens_evaluated": response.get("tokens_evaluated"),
        },
        "token_probability": {
            "availability": "available" if completion_probabilities else "capability_absent",
            "completion_probability_count": len(completion_probabilities),
            "top_logprob_row_count": sum(
                len(row.get("top_logprobs") or []) for row in completion_probabilities
            ),
            "probability_checksum": sha16(_stable_json(completion_probabilities))
            if completion_probabilities
            else None,
        },
        "completion_probabilities": completion_probabilities,
        "logits": {
            "availability": "available" if _surface_available(logits) else "capability_absent",
            "top_logits": logits.get("top_logits", []) if isinstance(logits, Mapping) else [],
        },
        "attention": {
            "availability": "available" if _surface_available(attention) else "capability_absent",
            "summary": attention,
        },
        "hidden_state_proxy": {
            "availability": "available"
            if _surface_available(hidden_proxy)
            else "capability_absent",
            "summary": hidden_proxy,
        },
        "token_timing": {
            "availability": "available" if token_timing_available else "capability_absent",
            "timings": dict(timings),
            "per_token_timings_count": len(raw_receipt.get("per_token_timings") or []),
        },
        "metadata_endpoint_receipts": raw_receipt.get("metadata_endpoint_receipts") or {},
        "runtime_error": raw_receipt.get("runtime_error"),
    }


def signal_availability(signal_receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "logits_available": (signal_receipt.get("logits") or {}).get("availability") == "available",
        "token_probability_available": (signal_receipt.get("token_probability") or {}).get(
            "availability"
        )
        == "available",
        "attention_available": (signal_receipt.get("attention") or {}).get("availability")
        == "available",
        "hidden_state_proxy_available": (signal_receipt.get("hidden_state_proxy") or {}).get(
            "availability"
        )
        == "available",
        "token_timing_available": (signal_receipt.get("token_timing") or {}).get("availability")
        == "available",
        "raw_output_receipt_available": (signal_receipt.get("raw_output") or {}).get("availability")
        == "available",
    }


def _receipt_kind(availability: Mapping[str, Any]) -> str:
    active = [field for field in CORE_SIGNAL_FIELDS if availability.get(field)]
    if len(active) > 1:
        return "multi_internal_signal"
    if active == ["logits_available"]:
        return "logits"
    if active == ["token_probability_available"]:
        return "token_probability"
    if active == ["attention_available"]:
        return "attention"
    if active == ["hidden_state_proxy_available"]:
        return "hidden_state_proxy"
    return "none"


def _missing_features(availability: Mapping[str, Any]) -> list[str]:
    names = {
        "logits_available": "logits_unavailable",
        "token_probability_available": "token_probability_metadata_unavailable",
        "attention_available": "attention_export_unavailable",
        "hidden_state_proxy_available": "hidden_state_proxy_unavailable",
        "token_timing_available": "token_timing_unavailable",
        "raw_output_receipt_available": "raw_output_receipt_unavailable",
    }
    return [missing for field, missing in names.items() if not availability.get(field)]


def _build_tiny_receipt(
    *,
    signal_receipt: Mapping[str, Any],
    availability: Mapping[str, Any],
    selected_model_spec: Mapping[str, Any] | None,
    selected_backend_command: Mapping[str, Any] | None,
) -> JsonDict | None:
    if not any(availability.get(field) for field in CORE_SIGNAL_FIELDS):
        return None
    return {
        "schema": TINY_RECEIPT_SCHEMA,
        "receipt_kind": _receipt_kind(availability),
        "model_role": (selected_model_spec or {}).get("role"),
        "model_hf_id": (selected_model_spec or {}).get("hf_id"),
        "model_path": (selected_model_spec or {}).get("model_path"),
        "backend_kind": signal_receipt.get("backend_kind")
        or (selected_backend_command or {}).get("backend_kind"),
        "endpoint": signal_receipt.get("endpoint"),
        "prompt_checksum": signal_receipt.get("prompt_checksum"),
        "raw_output": signal_receipt.get("raw_output"),
        "completion_probabilities": signal_receipt.get("completion_probabilities", []),
        "token_probability": signal_receipt.get("token_probability"),
        "logits": signal_receipt.get("logits"),
        "attention": signal_receipt.get("attention"),
        "hidden_state_proxy": signal_receipt.get("hidden_state_proxy"),
        "token_timing": signal_receipt.get("token_timing"),
        "quality_interpretation": None,
    }


def _schema_description(
    *,
    availability: Mapping[str, Any],
    ready: bool,
    tiny_receipt_path: Path | None,
    missing_features: Sequence[str],
) -> JsonDict:
    return {
        "schema": RECEIPT_SCHEMA,
        "internal_signal_receipt_ready": ready,
        "receipt_path": str(tiny_receipt_path) if ready and tiny_receipt_path else None,
        "receipt_kind": _receipt_kind(availability),
        "receipt_fields": [
            "model_role",
            "model_hf_id",
            "backend_kind",
            "endpoint",
            "prompt_checksum",
            "raw_output.content_checksum",
            "completion_probabilities",
            "token_probability",
            "logits",
            "attention",
            "hidden_state_proxy",
            "token_timing",
        ],
        "ready_condition": (
            "At least one reproducible local non-text internal field is present: "
            "token probabilities/logprobs, logits, attention, or hidden-state proxy data."
        ),
        "availability": dict(availability),
        "missing_backend_features": list(missing_features),
        "no_quality_claim": True,
        "external_text_scorer_reopened": False,
    }


def _precondition_blockers(
    *,
    prior_stable: bool,
    selected_model_spec: Mapping[str, Any] | None,
    selected_backend_command: Mapping[str, Any] | None,
    preconditions: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not prior_stable or selected_backend_command is None:
        blockers.append("exp5324_stable_backend_unavailable")
    if selected_model_spec is None:
        blockers.append("selected_mandated_model_unavailable")
    elif not _selected_model_file_present(selected_model_spec):
        blockers.append("selected_model_file_missing")
    if not preconditions.get("gpu_visible"):
        blockers.append("gpu_not_visible")
    if selected_backend_command is not None:
        command = selected_backend_command.get("command")
        if not isinstance(command, list) or not command:
            blockers.append("selected_backend_command_malformed")
        elif not Path(str(command[0])).is_file():
            blockers.append("selected_backend_binary_missing")
    return list(dict.fromkeys(blockers))


def default_native_server_signal_probe(
    *,
    selected_model_spec: Mapping[str, Any],
    selected_backend_command: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    prompt: str = PROMPT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> JsonDict:  # pragma: no cover
    _ = selected_backend_command
    server = (preconditions.get("binary_paths") or {}).get("llama-server")
    if not server or not Path(str(server)).is_file():
        return {
            "status": "blocked_llama_server_binary_missing",
            "backend_kind": "llama-server",
            "response_json": {},
            "runtime_error": "llama-server binary missing",
        }
    port = _free_port()
    command = [
        str(server),
        "-m",
        str(selected_model_spec["model_path"]),
        "-c",
        "512",
        "-b",
        "512",
        "-ub",
        "128",
        "-ngl",
        "all",
        "-sm",
        "layer",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "-np",
        "1",
        "--metrics",
        "--props",
        "--slots",
    ]
    started = time.perf_counter()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    response_json: JsonDict = {}
    metadata_receipts: JsonDict = {}
    runtime_error: str | None = None
    try:
        ready = _wait_for_health(port, min(timeout_s, 180.0))
        if not ready:
            runtime_error = "llama-server health endpoint did not become ready"
        else:
            response_json = _post_completion(port, prompt, timeout_s=max(1.0, timeout_s / 2.0))
            metadata_receipts = _collect_metadata_endpoints(port)
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    stderr_tail = proc.stderr.read()[-4000:] if proc.stderr else ""
    stdout_tail = proc.stdout.read()[-1000:] if proc.stdout else ""
    return {
        "status": "completed"
        if response_json and runtime_error is None
        else "blocked_probe_failed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "prompt": prompt,
        "command": command,
        "response_json": response_json,
        "metadata_endpoint_receipts": metadata_receipts,
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "runtime_error": runtime_error,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def _free_port() -> int:  # pragma: no cover
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, timeout_s: float) -> bool:  # pragma: no cover
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib_request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib_error.URLError, TimeoutError, OSError):
            time.sleep(1.0)
    return False


def _post_completion(port: int, prompt: str, timeout_s: float) -> JsonDict:  # pragma: no cover
    payload = {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0,
        "cache_prompt": False,
        "n_probs": 5,
    }
    req = urllib_request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", "replace")
    data = json.loads(body)
    return {
        "content": data.get("content", ""),
        "tokens_predicted": data.get("tokens_predicted"),
        "tokens_evaluated": data.get("tokens_evaluated"),
        "timings": data.get("timings"),
        "completion_probabilities": data.get("completion_probabilities") or [],
    }


def _collect_metadata_endpoints(port: int) -> JsonDict:  # pragma: no cover
    receipts: JsonDict = {}
    for endpoint in ("props", "slots", "metrics"):
        try:
            with urllib_request.urlopen(
                f"http://127.0.0.1:{port}/{endpoint}", timeout=10.0
            ) as resp:
                text = resp.read().decode("utf-8", "replace")
            receipts[endpoint] = {
                "ok": True,
                "status": resp.status,
                "body_checksum": sha16(text),
                "body_preview": text[:300],
            }
        except Exception as exc:
            receipts[endpoint] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return receipts


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    prior_artifact_path: Path | None = None,
    receipt_schema_path: Path | None = None,
    tiny_receipt_path: Path | None = None,
    preconditions_provider: PreconditionsProvider | None = None,
    option_surface_provider: OptionSurfaceProvider = default_backend_option_surface,
    signal_probe: SignalProbe = default_native_server_signal_probe,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    prior_artifact_path = prior_artifact_path or root / exp5324.RESULT_RELATIVE_PATH
    receipt_schema_path = receipt_schema_path or root / RECEIPT_SCHEMA_RELATIVE_PATH
    tiny_receipt_path = tiny_receipt_path or root / TINY_RECEIPT_RELATIVE_PATH

    prior_artifact = _read_json(prior_artifact_path)
    prior_stable = _prior_is_stable(prior_artifact)
    model_specs = _model_specs_from_prior(prior_artifact)
    selected_model_spec = _selected_model_from_prior(prior_artifact)
    selected_backend_command = _selected_command_from_prior(prior_artifact)
    preconditions_provider = preconditions_provider or (lambda: collect_current_preconditions(root))
    preconditions = dict(preconditions_provider())
    preconditions.update(
        {
            "exp5324_artifact_path": str(prior_artifact_path),
            "exp5324_stable": prior_stable,
            "selected_model_file_present": _selected_model_file_present(selected_model_spec),
            "selected_backend_kind": (selected_backend_command or {}).get("backend_kind"),
            "external_text_scorer_reopened": False,
        }
    )
    blockers = _precondition_blockers(
        prior_stable=prior_stable,
        selected_model_spec=selected_model_spec,
        selected_backend_command=selected_backend_command,
        preconditions=preconditions,
    )

    option_surface: JsonDict = {}
    signal_receipt = normalise_signal_receipt({}, prompt=PROMPT, backend_kind="not_probed")
    if not blockers:
        option_surface = dict(option_surface_provider(preconditions))
        preconditions["backend_option_surface_checked"] = True
        raw_signal = signal_probe(
            selected_model_spec=selected_model_spec,
            selected_backend_command=selected_backend_command,
            preconditions=preconditions,
            prompt=PROMPT,
        )
        signal_receipt = normalise_signal_receipt(
            raw_signal,
            prompt=PROMPT,
            backend_kind=str((selected_backend_command or {}).get("backend_kind") or "unknown"),
        )
    else:
        preconditions["backend_option_surface_checked"] = False
    preconditions["blocked_preconditions"] = blockers

    availability = signal_availability(signal_receipt)
    ready = bool(not blockers and any(availability.get(field) for field in CORE_SIGNAL_FIELDS))
    missing_features = _missing_features(availability)
    tiny_receipt = _build_tiny_receipt(
        signal_receipt=signal_receipt,
        availability=availability,
        selected_model_spec=selected_model_spec,
        selected_backend_command=selected_backend_command,
    )
    schema_description = _schema_description(
        availability=availability,
        ready=ready,
        tiny_receipt_path=tiny_receipt_path if tiny_receipt is not None else None,
        missing_features=missing_features,
    )

    status = "complete" if ready else "blocked"
    if ready:
        honest = f"complete: {_receipt_kind(availability)}_receipt_ready"
    elif blockers:
        honest = "blocked_preconditions:" + ",".join(blockers)
    else:
        honest = "blocked_internal_signal_unavailable:" + ",".join(missing_features)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "preconditions_checked": _wrap("preconditions_checked", preconditions),
        "selected_model_spec": _wrap("selected_model_spec", selected_model_spec),
        "receipt_schema_path": _wrap("receipt_schema_path", str(receipt_schema_path)),
        "tests_run": _wrap("tests_run", list(tests_run or [])),
        "backend_option_surface": _wrap("backend_option_surface", option_surface),
        "raw_output_receipt": _wrap("raw_output_receipt", signal_receipt.get("raw_output")),
        "missing_backend_features": _wrap("missing_backend_features", missing_features),
        "tiny_receipt_path": _wrap(
            "tiny_receipt_path", str(tiny_receipt_path) if tiny_receipt is not None else None
        ),
        "logits_available": availability["logits_available"],
        "token_probability_available": availability["token_probability_available"],
        "attention_available": availability["attention_available"],
        "hidden_state_proxy_available": availability["hidden_state_proxy_available"],
        "token_timing_available": availability["token_timing_available"],
        "raw_output_receipt_available": availability["raw_output_receipt_available"],
        "external_text_scorer_reopened": False,
        "internal_signal_receipt_ready": ready,
        "no_quality_claim": True,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "reproducibility_checksum": sha16(
            _stable_json(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "selected_model": selected_model_spec,
                    "availability": availability,
                    "ready": ready,
                    "missing_features": missing_features,
                    "seed": RANDOM_SEED,
                }
            )
        ),
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    if write:
        _write_json(receipt_schema_path, schema_description)
        if tiny_receipt is not None:
            _write_json(tiny_receipt_path, tiny_receipt)
        _write_json(artifact_path, artifact)
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

    for field in (
        "logits_available",
        "token_probability_available",
        "attention_available",
        "hidden_state_proxy_available",
        "external_text_scorer_reopened",
        "internal_signal_receipt_ready",
        "no_quality_claim",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare boolean")
    if artifact.get("external_text_scorer_reopened") is not False:
        errors.append("external_text_scorer_reopened must be bare false")
    if artifact.get("no_quality_claim") is not True:
        errors.append("no_quality_claim must be bare true")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        expected = {str(spec["role"]): str(spec["hf_id"]) for spec in MODEL_SPECS}
        if set(model_specs) != set(expected):
            errors.append("MODEL_SPECS roles mismatch")
        for role, hf_id in expected.items():
            row = model_specs.get(role)
            if not isinstance(row, Mapping) or row.get("hf_id") != hf_id:
                errors.append(f"MODEL_SPECS hf_id mismatch for {role}")
            elif row.get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")

    ready = artifact.get("internal_signal_receipt_ready")
    core_ready = any(bool(artifact.get(field)) for field in CORE_SIGNAL_FIELDS)
    if ready is True:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("ready artifact must have complete status")
        if not core_ready:
            errors.append("ready artifact must expose a non-text internal signal")
        if _wrapped_value(artifact, "tiny_receipt_path") in (None, ""):
            errors.append("ready artifact must include tiny_receipt_path")
    if ready is False and _wrapped_value(artifact, "status") != "blocked":
        errors.append("not-ready artifact must have blocked status")
    if not isinstance(_wrapped_value(artifact, "receipt_schema_path"), str):
        errors.append("receipt_schema_path must be a principle-wrapped path string")
    tests_run = _wrapped_value(artifact, "tests_run")
    if tests_run is not MISSING_WRAPPED_VALUE and not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--prior", type=Path, default=REPO_ROOT / exp5324.RESULT_RELATIVE_PATH)
    parser.add_argument("--schema-out", type=Path, default=REPO_ROOT / RECEIPT_SCHEMA_RELATIVE_PATH)
    parser.add_argument("--receipt-out", type=Path, default=REPO_ROOT / TINY_RECEIPT_RELATIVE_PATH)
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        prior_artifact_path=args.prior,
        receipt_schema_path=args.schema_out,
        tiny_receipt_path=args.receipt_out,
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5331] status={artifact['status']['value']} "
        f"internal_signal_receipt_ready={artifact['internal_signal_receipt_ready']} "
        f"out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
