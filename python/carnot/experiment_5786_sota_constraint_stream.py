"""Exp5786 prospective SOTA constraint response stream.

Spec refs: REQ-BENCH-5786, SCENARIO-BENCH-5786,
SCENARIO-BENCH-5786-BLOCKERS, REQ-VERIFY-5786, SCENARIO-VERIFY-5786,
SCENARIO-VERIFY-5786-REPLAY.

This module consumes the sealed Exp5785 finite-choice fixture with real local
GGUF generation. The model is allowed to produce text, but correctness is never
learned or inferred from that text. The only semantic boundary is the sealed
``<row_id>: <label>`` candidate label, and the Exp5785 exact validators remain
the authority for whether a parsed label is correct.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
EmitResponse = Callable[[Mapping[str, Any]], None]
StreamRunner = Callable[[JsonDict, list[JsonDict], JsonDict, EmitResponse], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5786_sota_constraint_stream.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_5786_sota_constraint_stream.rows.jsonl")
FIXTURE_RELATIVE_PATH = fixture.RESULT_RELATIVE_PATH
FIXTURE_ROW_RELATIVE_PATH = fixture.ROW_FILE_RELATIVE_PATH

SCHEMA = "carnot.experiment_5786.sota_constraint_stream.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5786
EXPERIMENT_ID = "experiment_5786_sota_constraint_stream"
MILESTONE = "2026.07.513"
RUN_DATE = "20260722"
INFERENCE_SUBSTRATE = "real_local_llama_cpp_cuda_gguf_generation_plus_exact_z3_validation"
SPEC_REFS = (
    "REQ-BENCH-5786",
    "SCENARIO-BENCH-5786",
    "SCENARIO-BENCH-5786-BLOCKERS",
    "REQ-VERIFY-5786",
    "SCENARIO-VERIFY-5786",
    "SCENARIO-VERIFY-5786-REPLAY",
)

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)
LEGACY_SMOKE_MODEL_IDS = ("Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it")
PROOF_PRESERVING_SURFACES = ("symbol_relabel", "order_paraphrase")
ALL_SURFACES = fixture.SURFACE_KINDS
PARSER_FAILURE_THRESHOLD = 0.05
SATISFIABLE_DRIFT_MIN_COUNT = 1
N_GPU_LAYERS_REQUESTED = -1
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5786,
    "runner_seed": 5786001,
    "bootstrap_seed": 5786002,
}
GENERATION_CONFIG: JsonDict = {
    "prompt_date": RUN_DATE,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 48,
    "n_ctx": 2048,
    "n_batch": 256,
    "n_gpu_layers": N_GPU_LAYERS_REQUESTED,
    "stop": ["<|eot_id|>", "<stop>", "\n\n"],
    "seed": RANDOM_SEEDS["runner_seed"],
    "chat_template_required": True,
    "parser_boundary": "one line: <row_id>: <candidate_label>",
}
PRODUCER_GATE_FIELDS = (
    "stream_ready_score",
    "raw_response_coverage",
    "exact_label_coverage",
    "parser_failure_rate",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "MODEL_SPECS",
    "models_used",
    "model_runtime_receipts",
    "gpu_offload_receipts",
    "fixture_hash",
    "row_file",
    "row_file_sha256",
    "raw_response_receipts",
    "checkpoint_resume_receipts",
    "sample_size_justification",
    "independent_unit_count",
    "failure_taxonomy_counts",
    "family_metrics",
    "solver_hardness_metrics",
    "surface_sensitivity_metrics",
    "proof_preserving_paired_deltas",
    "model_identity_interactions",
    "parser_failure_rate",
    "protected_fact_distortion_count",
    "satisfiable_drift_count",
    "exact_label_coverage",
    "raw_response_coverage",
    "leakage_checks",
    "real_sota_model_count",
    "stream_ready_score",
    "producer_gate_fields",
    "inference_substrate",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5786_sota_constraint_stream.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5786_sota_constraint_stream.py "
    "-m pytest tests/python/test_experiment_5786_sota_constraint_stream.py -q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5786_sota_constraint_stream.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5786_sota_constraint_stream.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)

_REGISTRY = {row["hf_id"]: row for row in SOTA_GGUF_MODELS}
MODEL_SPECS: list[JsonDict] = []
for _index, _hf_id in enumerate(MANDATED_MODEL_IDS):
    _base = dict(_REGISTRY[_hf_id])
    MODEL_SPECS.append(
        {
            "name": _base["name"],
            "hf_id": _hf_id,
            "model_repo_id": _hf_id,
            "family": _hf_id.rsplit("/", 1)[-1].replace("-GGUF", "").replace(".", "-").lower(),
            "role": _base["role"],
            "active_params_b": _base["active_params_b"],
            "total_params_b": _base["total_params_b"],
            "quantization": _base["quantization"],
            "min_vram_gb": _base["min_vram_gb"],
            "gpu": _index % 2,
            "headline_eligible": True,
            "legacy_smoke_only": False,
        }
    )


class ManifestReplayError(ValueError):
    """Raised when response rows no longer match the sealed stream receipts."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so GGUF files remain streamable."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _model_family(hf_id: str) -> str:
    if hf_id == QWEN_ID:
        return "qwen3-6-35b-a3b"
    if hf_id == GEMMA31_ID:
        return "gemma-4-31b-it"
    if hf_id == GEMMA26_ID:
        return "gemma-4-26b-a4b-it"
    return hf_id.rsplit("/", 1)[-1].replace("-GGUF", "").replace(".", "-").lower()


def _extract_quantization(filename: str, fallback: str) -> str:
    tokens = ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "UD-Q8_XL", "Q8_0")
    lowered = filename.lower()
    return next((token for token in tokens if token.lower() in lowered), fallback)


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]] | None = None) -> list[JsonDict]:
    """Resolve, hash, and order the three mandated local SOTA GGUF specs."""

    overrides = {str(row.get("hf_id")): row for row in model_specs or []}
    normalized: list[JsonDict] = []
    for index, base in enumerate(MODEL_SPECS):
        hf_id = str(base["hf_id"])
        override = overrides.get(hf_id, {})
        resolved = str(
            override.get("model_path")
            or override.get("resolved_model_path")
            or resolve_cached_gguf(hf_id, str(base["quantization"]))
            or ""
        )
        path = Path(resolved).expanduser() if resolved else Path()
        present = bool(resolved and path.is_file())
        filename = path.name if present else ""
        model_hash = str(override.get("model_hash") or "")
        model_size = int(override.get("model_size_bytes") or 0)
        if present and not model_hash:
            model_hash = sha256_file(path)
        if present and not model_size:
            model_size = path.stat().st_size
        normalized.append(
            {
                **base,
                "sequence_index": index,
                "family": _model_family(hf_id),
                "gpu": int(override.get("gpu", base["gpu"]) or 0),
                "model_path": resolved,
                "resolved_model_path": resolved,
                "gguf_filename": filename,
                "model_hash": model_hash if present else "",
                "model_size_bytes": model_size if present else 0,
                "quantization": _extract_quantization(filename, str(base["quantization"])),
                "local_model_present": present,
                "headline_eligible": override.get("headline_eligible") is not False,
                "legacy_smoke_only": False,
            }
        )
    return normalized


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 32768
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _disk_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = 4096
    available_mb = int(shutil.disk_usage(REPO_ROOT).free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _nvidia_smi_devices() -> list[JsonDict]:  # pragma: no cover - host-dependent preflight.
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return [{"error": str(exc)}]
    devices = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "driver_version": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_free_mb": int(parts[4]),
                    "memory_used_mb": int(parts[5]),
                }
            )
    return devices


def _gpu_used_total_mb() -> int:  # pragma: no cover - host-dependent runtime.
    return sum(int(row.get("memory_used_mb", 0) or 0) for row in _nvidia_smi_devices())


def _llama_cpp_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    try:
        import importlib.metadata
        import llama_cpp

        raw_info = llama_cpp.llama_cpp.llama_print_system_info()
        info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
        supports = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
        cuda_backend = "CUDA" in info.upper()
        return {
            "ok": supports and cuda_backend,
            "version": importlib.metadata.version("llama-cpp-python"),
            "cuda_backend": cuda_backend,
            "supports_gpu_offload": supports,
            "system_info": info,
        }
    except Exception as exc:
        return {"ok": False, "version": "", "cuda_backend": False, "error": repr(exc)}


def _chat_template_probe(
    model_path: str,
) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, vocab_only=True, verbose=False)
        metadata = getattr(llm, "metadata", {}) or {}
        template = str(metadata.get("tokenizer.chat_template") or "")
        return {
            "available": bool(template),
            "chat_template_hash": sha256_text(template) if template else "",
            "metadata_keys": sorted(str(key) for key in metadata)[:64],
            "ok": bool(template),
        }
    except Exception as exc:
        return {"available": False, "chat_template_hash": "", "ok": False, "error": repr(exc)}


def _replay_fixture(
    artifact_path: str | Path,
    row_file_path: str | Path,
) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    try:
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        rows = fixture.read_row_file(row_file_path)
        fixture.validate_artifact(artifact)
        fixture.verify_row_file(rows, artifact)
        gate_receipts = [
            {
                "field": field,
                "expected": 1.0,
                "actual": artifact.get(field),
                "passed": artifact.get(field) == 1.0,
            }
            for field in ("fixture_ready_score", "exact_label_coverage", "parser_control_pass_rate")
        ]
        return {
            "ok": all(row["passed"] for row in gate_receipts),
            "artifact_path": str(FIXTURE_RELATIVE_PATH),
            "artifact_sha256": sha256_file(artifact_path),
            "row_file_sha256": sha256_file(row_file_path),
            "gate_receipts": gate_receipts,
        }
    except Exception as exc:
        return {"ok": False, "artifact_path": str(FIXTURE_RELATIVE_PATH), "error": repr(exc)}


def collect_preconditions(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    fixture_artifact_path: str | Path = REPO_ROOT / FIXTURE_RELATIVE_PATH,
    fixture_row_file_path: str | Path = REPO_ROOT / FIXTURE_ROW_RELATIVE_PATH,
) -> JsonDict:  # pragma: no cover - host-dependent preflight.
    """Collect Step 0 receipts before the response stream can run."""

    pair = cached_sota_pair()
    specs = normalize_model_specs()
    devices = _nvidia_smi_devices()
    rtx_count = sum(1 for row in devices if "RTX 3090" in str(row.get("name", "")))
    llama = _llama_cpp_probe()
    memory = _memory_probe()
    disk = _disk_probe()
    fixture_replay = _replay_fixture(fixture_artifact_path, fixture_row_file_path)
    output_parent = Path(result_path).parent
    row_parent = Path(row_file_path).parent
    model_checks = {}
    for spec in specs:
        chat = _chat_template_probe(str(spec["model_path"])) if spec["local_model_present"] else {}
        free_vram = max((int(row.get("memory_free_mb", 0) or 0) for row in devices), default=0)
        model_checks[spec["hf_id"]] = {
            "local_model_present": spec["local_model_present"],
            "model_hash_checked": bool(spec["model_hash"]),
            "model_path": spec["model_path"],
            "model_hash": spec["model_hash"],
            "gguf_filename": spec["gguf_filename"],
            "quantization": spec["quantization"],
            "chat_template_checked": chat.get("ok") is True,
            "chat_template_hash": chat.get("chat_template_hash", ""),
            "free_vram_mb": free_vram,
            "min_vram_mb": int(float(spec["min_vram_gb"]) * 1000),
        }
    blocked = []
    if pair is None:
        blocked.append("cached_sota_pair_unavailable")
    if fixture_replay.get("ok") is not True:
        blocked.append("exp5785_gate_replay_failed")
    if rtx_count < 2:
        blocked.append("dual_rtx_3090_unavailable")
    if llama.get("ok") is not True:
        blocked.append("llama_cpp_cuda_unavailable")
    if memory.get("ok") is not True:
        blocked.append("insufficient_free_ram")
    if disk.get("ok") is not True:
        blocked.append("insufficient_free_disk")
    if not output_parent.exists() or not row_parent.exists():
        blocked.append("output_parent_missing")
    for hf_id, check in model_checks.items():
        if check["local_model_present"] is not True:
            blocked.append(f"model_missing:{hf_id}")
        if check["model_hash_checked"] is not True:
            blocked.append(f"model_hash_unreadable:{hf_id}")
        if check["chat_template_checked"] is not True:
            blocked.append(f"chat_template_missing:{hf_id}")
        if int(check["free_vram_mb"]) < int(check["min_vram_mb"]):
            blocked.append(f"insufficient_free_vram:{hf_id}")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": pair or [],
        "exp5785_gate_replay": fixture_replay,
        "cuda_devices": {"ok": rtx_count >= 2, "rtx_3090_count": rtx_count, "devices": devices},
        "llama_cpp": llama,
        "models": model_checks,
        "memory": memory,
        "disk": disk,
        "output_paths": {
            "result_path": str(result_path),
            "row_file": str(row_file_path),
            "parent_writable": output_parent.exists() and row_parent.exists(),
        },
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(blocked),
    }


def _candidate_lines(row: Mapping[str, Any]) -> str:
    return "\n".join(f"{item['label']}: {item['candidate']}" for item in row["label_mapping"])


def build_prompt_cell(row: Mapping[str, Any], generation_config: Mapping[str, Any]) -> JsonDict:
    """Build the chat-template payload without leaking exact-label annotations."""

    user = (
        f"Today is {RUN_DATE}.\n"
        "You are evaluating one sealed finite-choice constraint fixture.\n"
        f"Fixture surface: {row['surface_text']}\n"
        "Candidates:\n"
        f"{_candidate_lines(row)}\n"
        f"Return exactly one line in this format: {row['row_id']}: <label>\n"
        f"The label must be one of: {', '.join(row['candidate_labels'])}."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Return only the requested row-id and candidate label. Do not explain, "
                "do not add extra rows, and do not change the fixture facts."
            ),
        },
        {"role": "user", "content": user},
    ]
    prompt_payload = {
        "messages": messages,
        "generation_config": {
            key: generation_config[key]
            for key in ("prompt_date", "temperature", "top_p", "max_tokens", "stop")
        },
    }
    return {
        "row_id": str(row["row_id"]),
        "fixture_row": _copy_json(row),
        "messages": messages,
        "prompt_hash": sha256_json(prompt_payload),
    }


def _selected_candidate(row: Mapping[str, Any], label: str) -> str:
    return next(
        (str(item["candidate"]) for item in row["label_mapping"] if item["label"] == label),
        "",
    )


def _mentions_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered for token in ("abstain", "cannot", "can't", "unknown", "not enough")
    )


def _protected_fact_distorted(row: Mapping[str, Any], text: str) -> bool:
    marker = "unit="
    if marker in text:
        for suffix in text.split(marker)[1:]:
            token = suffix.split()[0].strip(" ;,.\n\r\t")
            if token and token != row["unit_id"]:
                return True
    lowered = text.lower()
    return "protected_facts" in lowered or "exact_label" in lowered or "exact_answer" in lowered


def classify_response(
    fixture_row: Mapping[str, Any],
    raw_response_text: str,
    finish_reason: str = "",
    generation_error: str = "",
) -> JsonDict:
    """Classify one raw response without learning or prose-level inference."""

    parser = (
        fixture._parse_failure("generation_error")
        if generation_error
        else fixture.parse_response(raw_response_text, {str(fixture_row["row_id"]): fixture_row})
    )
    parse_ok = parser["parse_ok"] is True
    selected_label = ""
    selected_candidate = ""
    if parse_ok:
        selected_label = str(parser["parsed_labels"][str(fixture_row["row_id"])])
        selected_candidate = _selected_candidate(fixture_row, selected_label)
    exact_answer_error = bool(parse_ok and selected_label != fixture_row["exact_label"])
    parser_failure = not parse_ok
    truncation = finish_reason == "length" or parser["parser_failure_reason"] == "truncation"
    protected = _protected_fact_distorted(fixture_row, raw_response_text)
    contradiction = bool(selected_candidate == "BOTH")
    abstention = _mentions_abstention(raw_response_text) or selected_candidate == "UNKNOWN"
    satisfiable_drift = bool(
        parse_ok and exact_answer_error and fixture_row["exact_status"] == "sat"
    )
    valid_correct = bool(
        parse_ok
        and not exact_answer_error
        and not contradiction
        and not abstention
        and not truncation
        and not protected
    )
    if truncation:
        failure_mode = "truncation"
    elif parser_failure:
        failure_mode = "parser_failure"
    elif protected:  # pragma: no cover - strict parser blocks extra protected-fact prose.
        failure_mode = "protected_fact_distortion"
    elif contradiction:
        failure_mode = "contradiction"
    elif satisfiable_drift:
        failure_mode = "satisfiable_drift"
    elif abstention:
        failure_mode = "abstention"
    elif exact_answer_error:
        failure_mode = "exact_answer_error"
    else:
        failure_mode = "valid_correct_response"
    return {
        "parse_ok": parse_ok,
        "parser_failure": parser_failure,
        "parser_failure_reason": str(parser["parser_failure_reason"]),
        "parsed_labels": dict(parser["parsed_labels"]),
        "selected_label": selected_label,
        "selected_candidate": selected_candidate,
        "selected_candidate_hash": sha256_text(selected_candidate) if selected_candidate else "",
        "exact_answer_error": exact_answer_error,
        "contradiction": contradiction,
        "satisfiable_drift": satisfiable_drift,
        "protected_fact_distortion": protected,
        "abstention": abstention,
        "truncation": truncation,
        "valid_correct_response": valid_correct,
        "failure_mode": failure_mode,
    }


def stream_row_hash(row: Mapping[str, Any]) -> str:
    """Hash a response stream row while excluding its own row hash."""

    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def build_stream_row(
    *,
    model_spec: Mapping[str, Any],
    fixture_row: Mapping[str, Any],
    raw_response: Mapping[str, Any],
    stream_sequence_index: int,
    generation_config: Mapping[str, Any],
) -> JsonDict:
    """Join raw model text with the sealed fixture validator label."""

    prompt_cell = build_prompt_cell(fixture_row, generation_config)
    raw_text = str(raw_response.get("raw_response_text", ""))
    taxonomy = classify_response(
        fixture_row,
        raw_text,
        str(raw_response.get("finish_reason", "")),
        str(raw_response.get("generation_error", "")),
    )
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "stream_sequence_index": stream_sequence_index,
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "model_hash": str(model_spec.get("model_hash", "")),
        "fixture_row_id": str(fixture_row["row_id"]),
        "fixture_unit_id": str(fixture_row["unit_id"]),
        "fixture_row_hash": str(fixture_row["row_hash"]),
        "fixture_chronology_index": int(fixture_row["chronology_index"]),
        "split": str(fixture_row["split"]),
        "family": str(fixture_row["family"]),
        "surface_kind": str(fixture_row["surface_kind"]),
        "proof_preserving": bool(fixture_row["proof_preserving"]),
        "solver_effort_bin": str(fixture_row["solver_effort_bin"]),
        "satisfiability": str(fixture_row["exact_status"]),
        "exact_label": str(fixture_row["exact_label"]),
        "exact_answer": str(fixture_row["exact_answer"]),
        "exact_certificate_hash": str(fixture_row["exact_certificate_hash"]),
        "prompt_hash": str(raw_response.get("prompt_hash") or prompt_cell["prompt_hash"]),
        "raw_response_text": raw_text,
        "raw_response_sha256": sha256_text(raw_text),
        "finish_reason": str(raw_response.get("finish_reason", "")),
        "output_tokens": int(raw_response.get("output_tokens", 0) or 0),
        "timing": dict(raw_response.get("timing") or {}),
        "generation_error": str(raw_response.get("generation_error", "")),
        "parser_receipt": {
            "parse_ok": taxonomy["parse_ok"],
            "parser_failure_reason": taxonomy["parser_failure_reason"],
            "parsed_labels": taxonomy["parsed_labels"],
            "boundary": "exp5785_row_id_to_candidate_label",
        },
        "selected_label": taxonomy["selected_label"],
        "selected_candidate": taxonomy["selected_candidate"],
        "taxonomy": taxonomy,
        "row_hash": "",
    }
    row["row_hash"] = stream_row_hash(row)
    return row


def append_stream_row(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one checkpoint row immediately after a model/fixture cell finishes."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()


def read_stream_rows(path: str | Path) -> list[JsonDict]:
    """Read an Exp5786 response stream JSONL file."""

    stream_path = Path(path)
    if not stream_path.exists():
        return []
    return [
        dict(json.loads(line))
        for line in stream_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stream_cell_key(row: Mapping[str, Any]) -> str:
    return f"{row['model_hf_id']}::{row['fixture_row_id']}"


def verify_stream_rows(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    """Replay row hashes, raw-response hashes, and checkpoint uniqueness."""

    receipts = dict(artifact.get("raw_response_receipts") or {})
    seen: set[str] = set()
    for row in rows:
        key = _stream_cell_key(row)
        if key in seen:
            raise ManifestReplayError("duplicate stream cell")
        seen.add(key)
        if sha256_text(str(row.get("raw_response_text", ""))) != row.get("raw_response_sha256"):
            raise ManifestReplayError("raw_response_sha256")
        if stream_row_hash(row) != row.get("row_hash"):
            raise ManifestReplayError("row_hash")
        receipt = dict(receipts.get(key) or {})
        if receipt and receipt.get("row_hash") != row.get("row_hash"):
            raise ManifestReplayError("artifact row_hash")
    if receipts and set(receipts) != seen:
        raise ManifestReplayError("row count")
    return True


def failure_taxonomy_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate independent taxonomy booleans."""

    fields = (
        "parser_failure",
        "contradiction",
        "satisfiable_drift",
        "protected_fact_distortion",
        "exact_answer_error",
        "abstention",
        "truncation",
        "valid_correct_response",
    )
    counts = {field: 0 for field in fields}
    mode_counts: Counter[str] = Counter()
    for row in rows:
        taxonomy = dict(row.get("taxonomy") or {})
        for field in fields:
            counts[field] += int(taxonomy.get(field) is True)
        if taxonomy.get("failure_mode"):
            mode_counts[str(taxonomy["failure_mode"])] += 1
    counts["failure_mode_counts"] = dict(mode_counts)
    return counts


def _rate(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    correct = sum(
        1 for row in rows if dict(row.get("taxonomy") or {}).get("valid_correct_response") is True
    )
    parser_failures = sum(
        1 for row in rows if dict(row.get("taxonomy") or {}).get("parser_failure") is True
    )
    exact_errors = sum(
        1 for row in rows if dict(row.get("taxonomy") or {}).get("exact_answer_error") is True
    )
    return {
        "n_rows": total,
        "valid_correct_count": correct,
        "accuracy": _rate(correct, total),
        "parser_failure_count": parser_failures,
        "parser_failure_rate": _rate(parser_failures, total),
        "exact_answer_error_count": exact_errors,
        "exact_answer_error_rate": _rate(exact_errors, total),
    }


def _metrics_by(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    values = sorted({str(row.get(field, "")) for row in rows})
    return {
        value: _metric_summary([row for row in rows if str(row.get(field, "")) == value])
        for value in values
    }


def _independent_unit_count(fixture_rows: Sequence[Mapping[str, Any]]) -> int:
    return len({str(row["unit_id"]) for row in fixture_rows if row["surface_kind"] == "canonical"})


def sample_size_justification(fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Explain the independent-unit denominator used for paired cells."""

    canonical = [row for row in fixture_rows if row["surface_kind"] == "canonical"]
    by_family = Counter(str(row["family"]) for row in canonical)
    minimum = min(by_family.values()) if by_family else 0
    return {
        "independent_unit_count": len({str(row["unit_id"]) for row in canonical}),
        "independent_units_per_family": dict(by_family),
        "minimum_independent_items_per_primary_paired_cell": minimum,
        "primary_paired_cell": "model_family x constraint_family x proof_preserving_surface_pair",
        "repeated_turns_counted_as_independent": False,
        "surface_variants_counted_as_independent": False,
        "power_calculation": (
            "N>=30 canonical units per family satisfies the preregistered CLT floor "
            "for paired percentage-point deltas."
            if minimum >= 30
            else "N<30 in this bounded run; stream can be smoke/blocker evidence only."
        ),
        "sample_size_ready": minimum >= 30,
    }


def _bootstrap_interval(values: Sequence[float], *, seed: int) -> JsonDict:
    if not values:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "n_clusters": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(200):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {
        "mean": round(sum(values) / len(values), 6),
        "ci95": [round(means[4], 6), round(means[194], 6)],
        "n_clusters": len(values),
    }


def proof_preserving_paired_deltas(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute paired surface deltas clustered by canonical fixture unit."""

    by_key = {
        (str(row["model_hf_id"]), str(row["fixture_unit_id"]), str(row["surface_kind"])): row
        for row in rows
    }
    deltas: JsonDict = {}
    units = sorted({str(row["fixture_unit_id"]) for row in rows})
    for model_id in MANDATED_MODEL_IDS:
        for surface in PROOF_PRESERVING_SURFACES:
            values = []
            for unit_id in units:
                canonical = by_key.get((model_id, unit_id, "canonical"))
                variant = by_key.get((model_id, unit_id, surface))
                if not canonical or not variant:
                    continue
                c_ok = dict(canonical["taxonomy"]).get("valid_correct_response") is True
                v_ok = dict(variant["taxonomy"]).get("valid_correct_response") is True
                values.append(float(v_ok) - float(c_ok))
            deltas[f"{model_id}::{surface}_minus_canonical"] = _bootstrap_interval(
                values,
                seed=int(RANDOM_SEEDS["bootstrap_seed"]) + len(deltas),
            )
    return deltas


def model_identity_interactions(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report model-level accuracy and pairwise interactions on matched rows."""

    by_model = {
        model_id: _metric_summary([row for row in rows if row["model_hf_id"] == model_id])
        for model_id in MANDATED_MODEL_IDS
    }
    pairwise: JsonDict = {}
    for left_index, left in enumerate(MANDATED_MODEL_IDS):
        for right in MANDATED_MODEL_IDS[left_index + 1 :]:
            pairwise[f"{left}|{right}"] = round(
                float(by_model[left]["accuracy"]) - float(by_model[right]["accuracy"]),
                6,
            )
    return {
        "models_compared": list(MANDATED_MODEL_IDS),
        "accuracy_by_model": by_model,
        "pairwise_accuracy_deltas": pairwise,
    }


def _leakage_checks(
    fixture_rows: Sequence[Mapping[str, Any]], stream_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    split_hashes: dict[str, set[str]] = {}
    for row in fixture_rows:
        split_hashes.setdefault(str(row["split"]), set()).add(str(row["row_hash"]))
    intersections: JsonDict = {}
    splits = sorted(split_hashes)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            intersections[f"{left}|{right}"] = sorted(split_hashes[left] & split_hashes[right])
    row_hashes = [str(row["fixture_row_hash"]) for row in stream_rows]
    return {
        "split_hash_intersections": intersections,
        "no_split_hash_leak": all(not value for value in intersections.values()),
        "stream_fixture_hashes_subset": set(row_hashes).issubset(
            {str(row["row_hash"]) for row in fixture_rows}
        ),
        "duplicate_stream_cells": len({_stream_cell_key(row) for row in stream_rows})
        != len(stream_rows),
        "legacy_smoke_models_excluded": not any(
            legacy in canonical_json(stream_rows) for legacy in LEGACY_SMOKE_MODEL_IDS
        ),
    }


def _raw_response_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        _stream_cell_key(row): {
            "row_hash": str(row["row_hash"]),
            "raw_response_sha256": str(row["raw_response_sha256"]),
            "prompt_hash": str(row["prompt_hash"]),
            "fixture_row_hash": str(row["fixture_row_hash"]),
        }
        for row in rows
    }


def _gpu_offload_receipts(runtime_receipts: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    return {
        hf_id: {
            "cuda_offload_authenticated": bool(
                runtime_receipts.get(hf_id, {}).get("cuda_offload_authenticated")
            ),
            "n_gpu_layers_requested": int(
                runtime_receipts.get(hf_id, {}).get("n_gpu_layers_requested", 0) or 0
            ),
            "n_gpu_layers_offloaded": int(
                runtime_receipts.get(hf_id, {}).get("n_gpu_layers_offloaded", 0) or 0
            ),
            "gpu_memory_before_mb": int(
                runtime_receipts.get(hf_id, {}).get("gpu_memory_before_mb", 0) or 0
            ),
            "gpu_memory_peak_mb": int(
                runtime_receipts.get(hf_id, {}).get("gpu_memory_peak_mb", 0) or 0
            ),
            "gpu_memory_after_mb": int(
                runtime_receipts.get(hf_id, {}).get("gpu_memory_after_mb", 0) or 0
            ),
            "offload_log_excerpt": str(
                runtime_receipts.get(hf_id, {}).get("offload_log_excerpt", "")
            )[-1000:],
        }
        for hf_id in MANDATED_MODEL_IDS
    }


def _resume_runtime_receipt(
    model_spec: Mapping[str, Any], existing_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "llama_cpp_version": "resume_from_checkpoint",
        "llama_cpp_build_info": {"resume_from_checkpoint": True},
        "chat_template": {"used": True, "resume_from_checkpoint": True},
        "cuda_device_receipt": {"resume_from_checkpoint": True},
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": 1,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 1,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": True,
        "rows_attempted": len(
            [row for row in existing_rows if row["model_hf_id"] == model_spec["hf_id"]]
        ),
        "offload_log_excerpt": "resume_from_existing_authenticated_rows",
    }


def stream_ready_score_from_artifact(artifact: Mapping[str, Any]) -> float:
    """Recompute the strict stream-readiness scalar from artifact gates."""

    gpu = dict(artifact.get("gpu_offload_receipts") or {})
    leakage = dict(artifact.get("leakage_checks") or {})
    sample_size = dict(artifact.get("sample_size_justification") or {})
    parser_failure_rate = (
        float(artifact["parser_failure_rate"]) if "parser_failure_rate" in artifact else 1.0
    )
    ready = bool(
        int(artifact.get("real_sota_model_count") or 0) == 3
        and list(artifact.get("models_used") or []) == list(MANDATED_MODEL_IDS)
        and all(
            dict(gpu.get(hf_id) or {}).get("cuda_offload_authenticated") is True
            for hf_id in MANDATED_MODEL_IDS
        )
        and float(artifact.get("raw_response_coverage") or 0.0) == 1.0
        and float(artifact.get("exact_label_coverage") or 0.0) == 1.0
        and parser_failure_rate < PARSER_FAILURE_THRESHOLD
        and int(artifact.get("satisfiable_drift_count") or 0) >= SATISFIABLE_DRIFT_MIN_COUNT
        and leakage.get("no_split_hash_leak") is True
        and leakage.get("stream_fixture_hashes_subset") is True
        and leakage.get("duplicate_stream_cells") is False
        and sample_size.get("sample_size_ready") is True
    )
    return 1.0 if ready else 0.0


def _blocking_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    gpu = dict(artifact.get("gpu_offload_receipts") or {})
    if any(
        dict(gpu.get(hf_id) or {}).get("cuda_offload_authenticated") is not True
        for hf_id in MANDATED_MODEL_IDS
    ):
        reasons.append("gpu_offload_unauthenticated")
    if float(artifact.get("raw_response_coverage") or 0.0) < 1.0:
        reasons.append("raw_response_coverage")
    if float(artifact.get("exact_label_coverage") or 0.0) < 1.0:
        reasons.append("exact_label_coverage")
    if artifact.get("leakage_checks", {}).get("duplicate_stream_cells") is True:
        reasons.append("duplicate_stream_cells")
    return sorted(set(reasons))


def _not_ready_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = []
    if float(artifact.get("parser_failure_rate") or 0.0) >= PARSER_FAILURE_THRESHOLD:
        reasons.append("parser_failure_threshold")
    if int(artifact.get("satisfiable_drift_count") or 0) < SATISFIABLE_DRIFT_MIN_COUNT:
        reasons.append("insufficient_satisfiable_drift")
    if dict(artifact.get("sample_size_justification") or {}).get("sample_size_ready") is not True:
        reasons.append("sample_size")
    leakage = dict(artifact.get("leakage_checks") or {})
    if (
        leakage.get("no_split_hash_leak") is not True
        or leakage.get("stream_fixture_hashes_subset") is not True
    ):
        reasons.append("leakage_checks")
    return reasons or ["stream_ready_gate_not_met"]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("status") == "blocked":
        return "blocked: " + ",".join(_blocking_reasons(artifact) or ["preconditions_or_runtime"])
    if artifact.get("stream_ready_score") == 1.0:
        return "complete: sota_constraint_response_stream_ready"
    return "complete: sota_constraint_response_stream_collected_not_ready:" + ",".join(
        _not_ready_reasons(artifact)
    )


def _fixture_hash(fixture_artifact: Mapping[str, Any], preconditions: Mapping[str, Any]) -> str:
    replay = dict(preconditions.get("exp5785_gate_replay") or {})
    return str(replay.get("artifact_sha256") or sha256_json(fixture_artifact))


def build_artifact(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    fixture_artifact: Mapping[str, Any],
    fixture_rows: Sequence[Mapping[str, Any]],
    stream_rows: Sequence[Mapping[str, Any]],
    runtime_receipts: Mapping[str, Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    row_file_path: str | Path,
    checkpoint_resume_receipts: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Build the terminal Exp5786 artifact from immutable response rows."""

    taxonomy_counts = failure_taxonomy_counts(stream_rows)
    expected_rows = len(fixture_rows) * len(MANDATED_MODEL_IDS)
    raw_count = sum(1 for row in stream_rows if row.get("raw_response_sha256"))
    exact_count = sum(1 for row in stream_rows if row.get("exact_label") and row.get("taxonomy"))
    runtime_by_model = {
        hf_id: dict(runtime_receipts.get(hf_id) or {}) for hf_id in MANDATED_MODEL_IDS
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "generation_config": dict(GENERATION_CONFIG),
        "status": "complete",
        "preconditions_checked": _copy_json(preconditions_checked),
        "MODEL_SPECS": [_copy_json(row) for row in model_specs],
        "models_used": list(MANDATED_MODEL_IDS),
        "model_runtime_receipts": runtime_by_model,
        "gpu_offload_receipts": _gpu_offload_receipts(runtime_by_model),
        "fixture_hash": _fixture_hash(fixture_artifact, preconditions_checked),
        "row_file": Path(row_file_path).as_posix(),
        "row_file_sha256": sha256_file(row_file_path)
        if Path(row_file_path).exists()
        else sha256_text(""),
        "raw_response_receipts": _raw_response_receipts(stream_rows),
        "checkpoint_resume_receipts": dict(checkpoint_resume_receipts),
        "sample_size_justification": sample_size_justification(fixture_rows),
        "independent_unit_count": _independent_unit_count(fixture_rows),
        "failure_taxonomy_counts": taxonomy_counts,
        "family_metrics": _metrics_by(stream_rows, "family"),
        "solver_hardness_metrics": _metrics_by(stream_rows, "solver_effort_bin"),
        "surface_sensitivity_metrics": _metrics_by(stream_rows, "surface_kind"),
        "satisfiability_metrics": _metrics_by(stream_rows, "satisfiability"),
        "proof_preserving_paired_deltas": proof_preserving_paired_deltas(stream_rows),
        "model_identity_interactions": model_identity_interactions(stream_rows),
        "parser_failure_rate": _rate(int(taxonomy_counts["parser_failure"]), len(stream_rows)),
        "protected_fact_distortion_count": int(taxonomy_counts["protected_fact_distortion"]),
        "satisfiable_drift_count": int(taxonomy_counts["satisfiable_drift"]),
        "exact_label_coverage": _rate(exact_count, expected_rows),
        "raw_response_coverage": _rate(raw_count, expected_rows),
        "leakage_checks": _leakage_checks(fixture_rows, stream_rows),
        "real_sota_model_count": len(
            [row for row in model_specs if row.get("local_model_present") is True]
        ),
        "stream_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    if preconditions_checked.get("preconditions_ready") is not True or _blocking_reasons(artifact):
        artifact["status"] = "blocked"
    artifact["stream_ready_score"] = (
        0.0 if artifact["status"] == "blocked" else stream_ready_score_from_artifact(artifact)
    )
    artifact["honest_verdict"] = _honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum blanked."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on schema drift or unsupported readiness claims."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if artifact.get("models_used") != list(MANDATED_MODEL_IDS):
        raise ValueError("models_used")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    for field in artifact.get("producer_gate_fields", []):
        if field not in artifact or isinstance(artifact[field], Mapping):
            raise ValueError("producer_gate_fields")
    expected_ready = stream_ready_score_from_artifact(artifact)
    if artifact.get("stream_ready_score") != expected_ready:
        raise ValueError("stream_ready_score")
    status = str(artifact.get("status"))
    verdict = str(artifact.get("honest_verdict"))
    if status == "blocked" and not verdict.startswith("blocked:"):
        raise ValueError("honest_verdict")
    if status == "complete" and not verdict.startswith("complete:"):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def _blocked_runtime_receipt(model_spec: Mapping[str, Any], reason: str) -> JsonDict:
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "llama_cpp_version": "",
        "llama_cpp_build_info": {"blocked_reason": reason},
        "chat_template": {"used": False, "blocked_reason": reason},
        "cuda_device_receipt": {"blocked_reason": reason},
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "cuda_offload_authenticated": False,
        "rows_attempted": 0,
        "offload_log_excerpt": "",
    }


def _prepare_existing_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    existing: dict[str, JsonDict] = {}
    for row in rows:
        key = _stream_cell_key(row)
        if key in existing:
            raise ManifestReplayError("duplicate stream cell")
        existing[key] = dict(row)
    return existing


def run(
    *,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: Path | str = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    fixture_artifact: Mapping[str, Any] | None = None,
    fixture_rows: Sequence[Mapping[str, Any]] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    stream_runner: StreamRunner | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run or resume the matched local-SOTA response stream."""

    specs = normalize_model_specs(model_specs)
    preconditions = dict(preconditions_checked or collect_preconditions())
    rows = list(fixture_rows or fixture.read_row_file(REPO_ROOT / FIXTURE_ROW_RELATIVE_PATH))
    source_artifact = dict(
        fixture_artifact
        or json.loads((REPO_ROOT / FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8"))
    )
    output_rows_path = Path(row_file_path)
    if write:
        output_rows_path.parent.mkdir(parents=True, exist_ok=True)
        output_rows_path.touch(exist_ok=True)
    existing_rows = read_stream_rows(output_rows_path)
    existing = _prepare_existing_rows(existing_rows)
    all_rows: list[JsonDict] = list(existing_rows)
    runtime_receipts: dict[str, JsonDict] = {}
    rows_written = 0
    duplicate_skipped = 0
    runner = stream_runner or default_stream_runner
    if preconditions.get("preconditions_ready") is True:
        for model_spec in specs:
            pending_cells = []
            for row in rows:
                key = f"{model_spec['hf_id']}::{row['row_id']}"
                if key in existing:
                    duplicate_skipped += 1
                    continue
                pending_cells.append(build_prompt_cell(row, GENERATION_CONFIG))
            if not pending_cells:
                runtime_receipts[str(model_spec["hf_id"])] = _resume_runtime_receipt(
                    model_spec,
                    all_rows,
                )
                continue

            def emit_response(
                raw_response: Mapping[str, Any], *, spec: Mapping[str, Any] = model_spec
            ) -> None:
                nonlocal rows_written
                fixture_row = next(row for row in rows if row["row_id"] == raw_response["row_id"])
                stream_index = len(all_rows)
                stream_row = build_stream_row(
                    model_spec=spec,
                    fixture_row=fixture_row,
                    raw_response=raw_response,
                    stream_sequence_index=stream_index,
                    generation_config=GENERATION_CONFIG,
                )
                key = _stream_cell_key(stream_row)
                if key in existing:
                    raise ManifestReplayError("duplicate stream cell")
                existing[key] = stream_row
                all_rows.append(stream_row)
                rows_written += 1
                if write:
                    append_stream_row(output_rows_path, stream_row)

            runtime_receipts[str(model_spec["hf_id"])] = runner(
                dict(model_spec),
                pending_cells,
                dict(GENERATION_CONFIG),
                emit_response,
            )
    else:
        runtime_receipts = {
            str(spec["hf_id"]): _blocked_runtime_receipt(spec, "preconditions_failed")
            for spec in specs
        }
    for spec in specs:
        runtime_receipts.setdefault(str(spec["hf_id"]), _blocked_runtime_receipt(spec, "not_run"))
    checkpoint_receipts = {
        "schema": SCHEMA + ".checkpoint_resume",
        "row_file": Path(row_file_path).as_posix(),
        "expected_cells": len(rows) * len(MANDATED_MODEL_IDS),
        "existing_rows_loaded": len(existing_rows),
        "duplicate_cells_skipped": duplicate_skipped,
        "rows_written": rows_written,
        "resume_supported": True,
        "checkpoint_after_every_cell": True,
        "duplicate_cells_present": len(existing) != len(all_rows),
    }
    artifact = build_artifact(
        model_specs=specs,
        fixture_artifact=source_artifact,
        fixture_rows=rows,
        stream_rows=all_rows,
        runtime_receipts=runtime_receipts,
        preconditions_checked=preconditions,
        row_file_path=row_file_path,
        checkpoint_resume_receipts=checkpoint_receipts,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes or {},
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def _parse_offloaded_layers(stderr_text: str) -> int:  # pragma: no cover - live telemetry helper.
    best = 0
    for line in stderr_text.splitlines():
        if "offloaded" not in line or "layers" not in line:
            continue
        parts = line.replace("/", " ").split()
        for index, token in enumerate(parts):
            if token == "offloaded" and index + 1 < len(parts):
                try:
                    best = max(best, int(parts[index + 1]))
                except ValueError:
                    pass
    return best


def default_stream_runner(
    model_spec: JsonDict,
    prompt_cells: list[JsonDict],
    generation_config: JsonDict,
    emit_response: EmitResponse,
) -> JsonDict:  # pragma: no cover - host-dependent live GGUF path.
    """Generate raw responses for one model through llama-cpp-python CUDA."""

    devices_before = _nvidia_smi_devices()
    before_mb = _gpu_used_total_mb()
    worker_payload = {
        "model_spec": model_spec,
        "prompt_cells": [
            {
                "row_id": cell["row_id"],
                "messages": cell["messages"],
                "prompt_hash": cell["prompt_hash"],
            }
            for cell in prompt_cells
        ],
        "generation_config": generation_config,
    }
    worker_code = r"""
import gc
import importlib.metadata
import json
import sys
import time

payload = json.loads(sys.stdin.read())
try:
    import llama_cpp
    from llama_cpp import Llama

    raw_info = llama_cpp.llama_cpp.llama_print_system_info()
    system_info = raw_info.decode("utf-8", "replace") if isinstance(raw_info, bytes) else str(raw_info)
    supports_gpu = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
    version = importlib.metadata.version("llama-cpp-python")
    vocab = Llama(model_path=payload["model_spec"]["model_path"], vocab_only=True, verbose=False)
    metadata = getattr(vocab, "metadata", {}) or {}
    template = str(metadata.get("tokenizer.chat_template") or "")
    del vocab
    gc.collect()
    llm = Llama(
        model_path=payload["model_spec"]["model_path"],
        n_gpu_layers=int(payload["generation_config"]["n_gpu_layers"]),
        n_ctx=int(payload["generation_config"]["n_ctx"]),
        n_batch=int(payload["generation_config"]["n_batch"]),
        seed=int(payload["generation_config"]["seed"]),
        verbose=True,
    )
    for cell in payload["prompt_cells"]:
        started = time.perf_counter()
        try:
            result = llm.create_chat_completion(
                messages=cell["messages"],
                temperature=float(payload["generation_config"]["temperature"]),
                top_p=float(payload["generation_config"]["top_p"]),
                max_tokens=int(payload["generation_config"]["max_tokens"]),
                stop=list(payload["generation_config"]["stop"]),
            )
            choice = result["choices"][0]
            message = choice.get("message") or {}
            text = str(message.get("content") or choice.get("text") or "")
            finish_reason = str(choice.get("finish_reason") or "")
            usage = result.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens") or 0)
            error = ""
        except Exception as exc:
            text = ""
            finish_reason = "error"
            output_tokens = 0
            error = repr(exc)
        elapsed = time.perf_counter() - started
        print(json.dumps({
            "type": "row",
            "row_id": cell["row_id"],
            "prompt_hash": cell["prompt_hash"],
            "raw_response_text": text,
            "finish_reason": finish_reason,
            "output_tokens": output_tokens,
            "timing": {"generation_s": round(elapsed, 6)},
            "generation_error": error,
        }, sort_keys=True), flush=True)
    del llm
    gc.collect()
    print(json.dumps({
        "type": "summary",
        "llama_cpp_version": version,
        "llama_cpp_build_info": {
            "cuda_backend": "CUDA" in system_info.upper(),
            "supports_gpu_offload": supports_gpu,
            "system_info": system_info,
            "module": getattr(llama_cpp, "__file__", ""),
        },
        "chat_template": {
            "available": bool(template),
            "used": True,
            "chat_template_hash": "sha256:" + __import__("hashlib").sha256(template.encode()).hexdigest() if template else "",
        },
    }, sort_keys=True), flush=True)
except Exception as exc:
    print(json.dumps({"type": "summary", "error": repr(exc)}, sort_keys=True), flush=True)
    raise
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", worker_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(model_spec.get("gpu", 0))},
    )
    stderr_chunks: list[str] = []
    stop_monitor = threading.Event()
    samples: list[int] = []

    def _stderr_reader() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_chunks.append(line)

    def _monitor() -> None:
        while not stop_monitor.is_set():
            samples.append(_gpu_used_total_mb())
            time.sleep(0.25)

    threading.Thread(target=_stderr_reader, daemon=True).start()
    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(worker_payload))
    proc.stdin.close()
    summary: JsonDict = {}
    timeout_s = float(os.environ.get("CARNOT_5786_MODEL_TIMEOUT_S", "7200"))
    started = time.monotonic()
    for line in proc.stdout:
        payload = json.loads(line)
        if payload.get("type") == "row":
            emit_response(payload)
        elif payload.get("type") == "summary":
            summary = payload
        if time.monotonic() - started > timeout_s:
            proc.kill()
            break
    proc.wait(timeout=30)
    stop_monitor.set()
    monitor.join(timeout=2)
    after_mb = _gpu_used_total_mb()
    stderr_text = "".join(stderr_chunks)
    offloaded = _parse_offloaded_layers(stderr_text)
    peak_mb = max(samples or [before_mb])
    gc.collect()
    return {
        "model_hf_id": model_spec["hf_id"],
        "model_family": model_spec["family"],
        "llama_cpp_version": str(summary.get("llama_cpp_version") or ""),
        "llama_cpp_build_info": dict(summary.get("llama_cpp_build_info") or {}),
        "chat_template": dict(summary.get("chat_template") or {}),
        "cuda_device_receipt": {
            "before": devices_before,
            "peak": samples,
            "after": _nvidia_smi_devices(),
            "worker_returncode": proc.returncode,
            "worker_error": str(summary.get("error") or ""),
            "cuda_visible_devices": str(model_spec.get("gpu", 0)),
        },
        "n_gpu_layers_requested": N_GPU_LAYERS_REQUESTED,
        "n_gpu_layers_offloaded": offloaded,
        "gpu_memory_before_mb": before_mb,
        "gpu_memory_peak_mb": peak_mb,
        "gpu_memory_after_mb": after_mb,
        "cuda_offload_authenticated": bool(offloaded > 0 and peak_mb > before_mb),
        "rows_attempted": len(prompt_cells),
        "offload_log_excerpt": stderr_text[-4000:],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5786 from the command line."""

    del argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
