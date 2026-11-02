"""Measure embedded-chat transport on three local SOTA GGUF families.

Spec refs: REQ-INFRA-7085, REQ-VERIFY-7085, and their scenarios.

The worker saves raw response bytes before the controller parses them. The
controller opens the exact Exp7064 entrance table only after all workers exit.
This keeps transport evidence separate from arithmetic labels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

from carnot import experiment_7064_v619_exact_entrance_fixture as fixture_api
from carnot import experiment_7079_v620_gpu_lease_audit as lease_audit_api
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_load_envelope_canary import (
    build_vram_release_row,
    embedded_tokenizer_probe,
    gpu_inventory,
    llama_cpp_probe,
    parse_offloaded_layers,
)
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.task_runtime_receipts import sha256_file, write_json_atomic


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "carnot.experiment_7085_v621_chat_transport_canary"
EXPERIMENT_ID = "experiment_7085_v621_chat_transport_canary"
SCHEMA = "carnot.experiment_7085.v621_chat_transport_canary.v1"
RUN_DATE = "20260906"
RANDOM_SEED = 7_085_202_609_06
GENERATION_SEED = 7_085_001
UNITS_PER_MODEL = 8
INFERENCE_SUBSTRATE = "bounded live local SOTA GGUF chat generation"
RESULT_PATH = REPO_ROOT / "results/experiment_7085_v621_chat_transport_canary.json"
FIXTURE_PATH = REPO_ROOT / "results/experiment_7064_v619_exact_entrance_fixture.json"
PINNED_FIXTURE_SHA256 = "sha256:6b62768e3387d40eebf462c199aab6a440321aa4a1549ff7d54faaba312f2277"
LEASE_AUDIT_PATH = REPO_ROOT / "results/experiment_7079_v620_gpu_lease_audit.json"
PINNED_LEASE_AUDIT_SHA256 = (
    "sha256:1c45ac780b3221d19179d8af0f25004efb73fe144539f944add4dedae705f32f"
)
EXP7080_PATH = REPO_ROOT / "results/experiment_7080_v620_three_family_entrance_bank.json"
EXP6200_PATH = REPO_ROOT / "results/experiment_6200_three_family_raw_code_transport_canary.json"
RAW_DIR = REPO_ROOT / "results/raw/experiment_7085_v621_chat_transport_canary"
CHECKPOINT_DIR = REPO_ROOT / "results/checkpoints/experiment_7085_v621_chat_transport_canary"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
PREFERRED_QUANT = "Q4_K_M"
MODEL_TIMEOUT_S = 3_600.0
VRAM_RELEASE_TIMEOUT_S = 180.0
VRAM_RELEASE_TOLERANCE_MB = 512

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

GENERATION_CONFIG: JsonDict = {
    "n_ctx": 2_048,
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_ubatch": 512,
    "main_gpu": 0,
    "split_mode": "layer",
    "tensor_split": [0.5, 0.5],
    "temperature": 0.2,
    "top_p": 0.9,
    "completion_budget_tokens": 192,
    "stop": [],
    "response_format": {"type": "json_object"},
    "visible_devices": [0, 1],
    "tokenizer_source": "embedded_gguf",
}

SYSTEM_MESSAGE = (
    "Follow the user instruction. Return only the requested compact JSON object. "
    "Do not include analysis, markdown, or extra keys."
)
PROMPT_TEMPLATE = (
    "Use the available positive integers exactly as a multiset. Combine two values with +, -, *, "
    "or exact integer /. Subtraction must stay positive. The target is {target}. Available integers: "
    "{numbers}. Propose only the first arithmetic branch. Return exactly one JSON object with this "
    'schema: {{"operand_pair":[smaller_integer,larger_integer],"operator":"+|-|*|/"}}. '
    "Do not solve later branches."
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "MODEL_SPECS",
    "inference_substrate",
    "inference_substrate_class",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "rows",
    "per_game_results",
    "model_specs",
    "model_identity_rows",
    "model_execution_rows",
    "chat_template_rows",
    "role_message_rows",
    "rendered_prompt_hash_rows",
    "stop_config_rows",
    "raw_output_rows",
    "token_count_rows",
    "finish_reason_rows",
    "parse_rows",
    "exact_label_rows",
    "per_model_rows",
    "model_order_rows",
    "sampling_config",
    "seed_rows",
    "checkpoint_rows",
    "runner_receipt",
    "generation_invoked",
    "total_model_count",
    "model_load_count_by_stage",
    "per_model_duration_s",
    "stage_gpu_telemetry_rows",
    "task_gpu_telemetry_rows",
    "peak_vram_by_device",
    "gpu_lease_rows",
    "vram_release_rows",
    "signals_sent",
    "empty_output_rate_by_model",
    "zero_token_rate_by_model",
    "parseable_rate_by_model",
    "leaked_control_token_count_by_model",
    "length_limited_count_by_model",
    "all_models_real",
    "chat_transport_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

_FIELD_REASONS = {
    "field_principles": "Field-level reasons expose omitted evidence.",
    "preconditions_checked": "Measured gates prevent fabricated execution.",
    "MODEL_SPECS": "The public roster prevents model substitution.",
    "inference_substrate": "The substrate states what generated the evidence.",
    "inference_substrate_class": "The class separates model work from a blocked run.",
    "duration_s": "Measured wall time exposes implausible execution.",
    "source_artifact_hashes": "Hashes bind results to reviewed inputs and code.",
    "cited_upstream_artifacts": "Citations identify every reused result.",
    "rows": "Gate rows make the verdict independently recomputable.",
    "per_game_results": "Unit summaries expose uneven failures.",
    "model_specs": "The internal roster must match the public roster.",
    "model_identity_rows": "Embedded metadata binds bytes to model identity.",
    "model_execution_rows": "Execution rows prove each family ran locally.",
    "chat_template_rows": "Template receipts prove instruction transport.",
    "role_message_rows": "Role rows preserve the chat turn contract.",
    "rendered_prompt_hash_rows": "Rendered hashes detect backend prompt drift when exposed.",
    "stop_config_rows": "Stop rows expose termination assumptions.",
    "raw_output_rows": "Raw bytes precede interpretation.",
    "token_count_rows": "Token counts detect immediate end-of-sequence failures.",
    "finish_reason_rows": "Finish reasons expose truncation.",
    "parse_rows": "Parse evidence stays separate from exact authority.",
    "exact_label_rows": "Exact labels measure legality without changing raw data.",
    "per_model_rows": "Family summaries prevent pooled credit.",
    "model_order_rows": "Order receipts reproduce resource sequencing.",
    "sampling_config": "A frozen sampler isolates transport and family.",
    "seed_rows": "A fixed seed reproduces generation and order.",
    "checkpoint_rows": "Checkpoints make interruption recovery auditable.",
    "runner_receipt": "Runner evidence identifies llama.cpp and CUDA transport.",
    "generation_invoked": "This flag distinguishes attempted generation from preflight.",
    "total_model_count": "The count detects a missing family.",
    "model_load_count_by_stage": "Load counts detect hidden reloads.",
    "per_model_duration_s": "Family durations expose missing or stalled work.",
    "stage_gpu_telemetry_rows": "Stage samples bind work to accelerator use.",
    "task_gpu_telemetry_rows": "Task samples retain the full resource boundary.",
    "peak_vram_by_device": "Peak memory authenticates model residence.",
    "gpu_lease_rows": "Lease history proves exclusive ownership.",
    "vram_release_rows": "Release evidence prevents cross-model contamination.",
    "signals_sent": "The signal ledger protects unrelated processes.",
    "empty_output_rate_by_model": "Empty rates detect failed response transport.",
    "zero_token_rate_by_model": "Zero-token rates detect immediate termination.",
    "parseable_rate_by_model": "Parse rates measure schema transport.",
    "leaked_control_token_count_by_model": "Leak counts detect template boundary failures.",
    "length_limited_count_by_model": "Length counts detect insufficient output budgets.",
    "all_models_real": "This gate excludes remote and legacy fallbacks.",
    "chat_transport_ready_score": "Readiness is one only when every family passes.",
    "random_seed": "The controller seed fixes all randomized choices.",
    "reproducibility_checksum": "The content digest detects artifact mutation.",
    "gate_check_summary": "Exact expected and observed values make blocks actionable.",
    "verifier_is_oracle": "False prevents exact labels from proving model quality.",
    "verdict_class": "A closed terminal class supports automation.",
    "honest_verdict": "A class prefix states the measured boundary.",
}
FIELD_PRINCIPLES = {field: _FIELD_REASONS[field] for field in REQUIRED_ARTIFACT_FIELDS}

CONTROL_TOKEN_RE = re.compile(
    r"<\|[^>]+\|>|<(?:/?(?:start|end)_of_turn|bos|eos)>|<channel\|>", re.IGNORECASE
)


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable keys and bytes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=lambda item: item.item(),
    )


def sha256_text(value: str) -> str:
    """Hash the exact UTF-8 bytes of text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one expected-observed schema for every gate."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve all checks and promote the first failure."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the mandated cached pair before the exact third family."""

    pair = cached_pair_func(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    pair_paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows = []
    for model_id in REQUIRED_MODEL_IDS:
        path = pair_paths.get(model_id) or resolver(model_id, PREFERRED_QUANT) or ""
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": path,
                "gpu_indices": [0, 1],
                "preferred_quant": PREFERRED_QUANT,
                "tokenizer_source": "embedded_gguf",
                "remote_allowed": False,
                "headline_eligible": True,
                "resolution_method": (
                    "cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 2))"
                    if model_id in pair_paths
                    else "cached_sota_pair exact-family cached extension"
                ),
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject cache misses, roster drift, and non-local model paths."""

    errors = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_ids_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id", ""))
        path = str(row.get("model_path", ""))
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif Path(path).suffix.lower() != ".gguf" or "mmproj" in Path(path).name.lower():
            errors.append(f"model_path_not_primary_gguf:{model_id}")
        if row.get("gpu_indices") != [0, 1]:
            errors.append(f"gpu_indices_mismatch:{model_id}")
        if row.get("tokenizer_source") != "embedded_gguf":
            errors.append(f"tokenizer_source_mismatch:{model_id}")
        if row.get("remote_allowed") is not False or row.get("headline_eligible") is not True:
            errors.append(f"headline_policy_mismatch:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def model_identity_errors(
    specs: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Require matching embedded identity and a non-empty chat template."""

    by_id = {str(row.get("model_id")): row for row in rows}
    errors = []
    for spec in specs:
        model_id = str(spec.get("hf_id"))
        row = by_id.get(model_id, {})
        if row.get("identity_matches") is not True:
            errors.append(f"model_identity_mismatch:{model_id}")
        if row.get("tokenizer_source") != "embedded_gguf":
            errors.append(f"model_tokenizer_mismatch:{model_id}")
        if row.get("chat_template_present") is not True or not row.get("chat_template_hash"):
            errors.append(f"model_chat_template_missing:{model_id}")
    return errors


def build_proposal_prompt(unit: Mapping[str, Any]) -> str:
    """Render only prompt-visible unit fields."""

    return PROMPT_TEMPLATE.format(
        target=int(unit["target"]),
        numbers=canonical_json([int(value) for value in unit["numbers"]]),
    )


def build_role_messages(prompt: str) -> list[JsonDict]:
    """Build the frozen instruction turn sequence."""

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": str(prompt)},
    ]


def select_representative_units(
    unit_rows: Sequence[Mapping[str, Any]], *, count: int = UNITS_PER_MODEL
) -> list[JsonDict]:
    """Select broad source coverage before taking second units."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        groups[str(row.get("source_group_id", ""))].append(row)
    selected: list[Mapping[str, Any]] = []
    depth = 0
    ordered_groups = sorted(groups)
    while len(selected) < count and any(depth < len(groups[group]) for group in ordered_groups):
        for group in ordered_groups:
            if depth < len(groups[group]) and len(selected) < count:
                selected.append(groups[group][depth])
        depth += 1
    if len(selected) != count:
        raise ValueError(f"representative_unit_count:{len(selected)}")
    return [deepcopy(dict(row)) for row in selected]


def build_schedule(
    specs: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build the matched model-by-unit transport schedule."""

    rows = []
    for spec in specs:
        for unit in units:
            prompt = build_proposal_prompt(unit)
            messages = build_role_messages(prompt)
            rows.append(
                {
                    "raw_key": f"{spec['hf_id']}|{unit['unit_id']}|{GENERATION_SEED}",
                    "model_id": str(spec["hf_id"]),
                    "model_path": str(spec["model_path"]),
                    "unit_id": str(unit["unit_id"]),
                    "source_group_id": str(unit.get("source_group_id", "")),
                    "seed": GENERATION_SEED,
                    "prompt": prompt,
                    "prompt_hash": sha256_text(prompt),
                    "role_messages": messages,
                    "role_messages_hash": sha256_text(canonical_json(messages)),
                    "generation_config": deepcopy(GENERATION_CONFIG),
                }
            )
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["execution_index"] = index
    return rows


def schedule_errors(rows: Sequence[Mapping[str, Any]], unit_ids: Sequence[str]) -> list[str]:
    """Check the frozen cross-family prompt and sampler contract."""

    errors = []
    expected = {(model_id, unit_id) for model_id in REQUIRED_MODEL_IDS for unit_id in unit_ids}
    observed = {(str(row.get("model_id")), str(row.get("unit_id"))) for row in rows}
    if observed != expected or len(rows) != len(expected):
        errors.append("schedule_key_set_mismatch")
    for row in rows:
        prompt = str(row.get("prompt", ""))
        if row.get("prompt_hash") != sha256_text(prompt):
            errors.append("prompt_hash_mismatch")
        expected_messages = build_role_messages(prompt)
        if row.get("role_messages") != expected_messages:
            errors.append("role_message_mismatch")
        if row.get("role_messages_hash") != sha256_text(canonical_json(row.get("role_messages"))):
            errors.append("role_message_hash_mismatch")
        config = dict(row.get("generation_config") or {})
        if config.get("stop") != GENERATION_CONFIG["stop"]:
            errors.append("stop_config_mismatch")
        if config != GENERATION_CONFIG:
            errors.append("sampling_config_mismatch")
        if row.get("seed") != GENERATION_SEED:
            errors.append("generation_seed_mismatch")
    for unit_id in unit_ids:
        unit_rows = [row for row in rows if row.get("unit_id") == unit_id]
        if len({str(row.get("prompt")) for row in unit_rows}) != 1:
            errors.append(f"cross_model_prompt_mismatch:{unit_id}")
    return list(dict.fromkeys(errors))


def randomized_model_order(seed: int = RANDOM_SEED) -> list[str]:
    """Return the reproducible model execution order."""

    rows = list(REQUIRED_MODEL_IDS)
    random.Random(int(seed)).shuffle(rows)
    return rows


def model_order_rows(seed: int = RANDOM_SEED) -> list[JsonDict]:
    """Expose model randomization as ordered evidence rows."""

    return [
        {"order_index": index, "model_id": model_id, "random_seed": int(seed)}
        for index, model_id in enumerate(randomized_model_order(seed))
    ]


def _template_text(metadata: Mapping[str, Any]) -> str:
    """Read the embedded template from known GGUF metadata keys."""

    for key in ("tokenizer.chat_template", "tokenizer.chat_template.default"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def render_chat_prompt_hash(
    template: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    eos_token: str = "",
    bos_token: str = "",
) -> str | None:
    """Hash the prompt with llama.cpp's own safe Jinja formatter."""

    if not template:
        return None
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        rendered = Jinja2ChatFormatter(
            template=template,
            eos_token=eos_token,
            bos_token=bos_token,
            add_generation_prompt=True,
        )(messages=[deepcopy(dict(row)) for row in messages])
        return sha256_text(rendered.prompt)
    except Exception:  # noqa: BLE001 - unavailable prompt rendering is recorded as null.
        return None


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Sync each raw row so parsing cannot outrun durable bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path, manifest_hash: str) -> list[JsonDict]:
    """Load only manifest-bound checkpoints with unique row keys."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("manifest_hash") != manifest_hash:
        raise ValueError("checkpoint_manifest_mismatch")
    rows = [deepcopy(dict(row)) for row in document.get("rows", [])]
    keys = [str(row.get("raw_key")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("checkpoint_duplicate_raw_key")
    return rows


def checkpoint_raw_row(path: Path, manifest_hash: str, row: Mapping[str, Any]) -> JsonDict:
    """Append one immutable raw row to an atomic checkpoint."""

    rows = load_checkpoint(path, manifest_hash) if path.is_file() else []
    key = str(row.get("raw_key"))
    existing = next((item for item in rows if item.get("raw_key") == key), None)
    if existing is not None:
        if existing != dict(row):
            raise ValueError("checkpoint_row_mismatch")
        return {"raw_key": key, "written": False, "path": str(path)}
    rows.append(deepcopy(dict(row)))
    write_json_atomic(path, {"manifest_hash": manifest_hash, "rows": rows})
    return {"raw_key": key, "written": True, "path": str(path)}


def worker_generate_one(
    schedule_row: Mapping[str, Any],
    *,
    llama_factory: Callable[..., Any] | None = None,
    llama_instance: Any = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Run one embedded-template chat call and retain its raw evidence."""

    owns_llama = llama_instance is None
    llm = llama_instance
    started_ns = clock()
    close_called = False
    try:
        config = dict(schedule_row.get("generation_config") or GENERATION_CONFIG)
        if llm is None:
            if llama_factory is None:  # pragma: no cover - live worker import.
                from llama_cpp import Llama

                llama_factory = Llama
            llm = llama_factory(
                model_path=str(schedule_row["model_path"]),
                n_ctx=int(config["n_ctx"]),
                n_gpu_layers=int(config["n_gpu_layers"]),
                n_batch=int(config["n_batch"]),
                n_ubatch=int(config["n_ubatch"]),
                main_gpu=int(config["main_gpu"]),
                split_mode=1,
                tensor_split=list(config["tensor_split"]),
                use_mmap=True,
                seed=int(schedule_row["seed"]),
                verbose=True,
            )
        metadata = deepcopy(dict(getattr(llm, "metadata", {}) or {}))
        template = _template_text(metadata)
        if not template:
            raise ValueError("missing_or_empty_embedded_chat_template")
        messages = deepcopy(list(schedule_row["role_messages"]))
        eos_token = ""
        bos_token = ""
        model = getattr(llm, "_model", None)
        if model is not None:
            eos_id = int(llm.token_eos())
            bos_id = int(llm.token_bos())
            eos_token = model.token_get_text(eos_id) if eos_id != -1 else ""
            bos_token = model.token_get_text(bos_id) if bos_id != -1 else ""
        rendered_hash = render_chat_prompt_hash(
            template,
            messages,
            eos_token=eos_token,
            bos_token=bos_token,
        )
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=int(config["completion_budget_tokens"]),
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            seed=int(schedule_row["seed"]),
            stop=deepcopy(config["stop"]),
            response_format=deepcopy(config["response_format"]),
        )
        ended_ns = clock()
        choice = (response.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        text = str(message.get("content") or "")
        usage = dict(response.get("usage") or {})
        row = {
            **deepcopy(dict(schedule_row)),
            "transport_method": "create_chat_completion",
            "chat_format": str(getattr(llm, "chat_format", None) or "embedded_gguf_template"),
            "chat_template_present": True,
            "chat_template_hash": sha256_text(template),
            "rendered_prompt_hash": rendered_hash,
            "rendered_prompt_hash_available": rendered_hash is not None,
            "stop_config": deepcopy(config["stop"]),
            "raw_text": text,
            "raw_bytes_hex": text.encode("utf-8").hex(),
            "raw_output_hash": sha256_text(text),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "finish_reason": str(choice.get("finish_reason") or "unknown"),
            "timings": {
                "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
                "backend": deepcopy(dict(response.get("timings") or {})),
            },
            "terminal_state": "complete",
            "exception_type": None,
            "exception_message": None,
            "raw_persisted_before_parse": True,
            "parsed_at_write_time": False,
            "labeled_at_write_time": False,
        }
    except Exception as exc:  # noqa: BLE001 - backend failure is evidence.
        ended_ns = clock()
        row = {
            **deepcopy(dict(schedule_row)),
            "transport_method": "create_chat_completion",
            "chat_format": str(getattr(llm, "chat_format", None) or "unknown")
            if llm
            else "unknown",
            "chat_template_present": False,
            "chat_template_hash": None,
            "rendered_prompt_hash": None,
            "rendered_prompt_hash_available": False,
            "stop_config": deepcopy(
                dict(schedule_row.get("generation_config") or GENERATION_CONFIG).get("stop", [])
            ),
            "raw_text": "",
            "raw_bytes_hex": "",
            "raw_output_hash": sha256_text(""),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": None,
            "timings": {
                "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
                "backend": {},
            },
            "terminal_state": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "exception_traceback": traceback.format_exc(),
            "raw_persisted_before_parse": True,
            "parsed_at_write_time": False,
            "labeled_at_write_time": False,
        }
    finally:
        if owns_llama and llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
                close_called = True
        if owns_llama:
            llm = None
            gc.collect()
    row["model_close_called"] = close_called if owns_llama else False
    return row


def worker_run_schedule(
    payload: Mapping[str, Any], *, llama_factory: Callable[..., Any] | None = None
) -> JsonDict:
    """Load one model once and checkpoint every missing unit."""

    if llama_factory is None:  # pragma: no cover - live worker import.
        from llama_cpp import Llama

        llama_factory = Llama
    config = dict(GENERATION_CONFIG)
    llm = llama_factory(
        model_path=str(payload["model_path"]),
        n_ctx=int(config["n_ctx"]),
        n_gpu_layers=int(config["n_gpu_layers"]),
        n_batch=int(config["n_batch"]),
        n_ubatch=int(config["n_ubatch"]),
        main_gpu=int(config["main_gpu"]),
        split_mode=1,
        tensor_split=list(config["tensor_split"]),
        use_mmap=True,
        seed=GENERATION_SEED,
        verbose=True,
    )
    raw_path = Path(str(payload["raw_path"]))
    checkpoint_path = Path(str(payload["checkpoint_path"]))
    manifest_hash = str(payload["manifest_hash"])
    existing = load_checkpoint(checkpoint_path, manifest_hash) if checkpoint_path.is_file() else []
    completed = {str(row["raw_key"]) for row in existing}
    receipts = []
    metadata = deepcopy(dict(getattr(llm, "metadata", {}) or {}))
    try:
        for schedule_row in payload["schedule_rows"]:
            if str(schedule_row["raw_key"]) in completed:
                continue
            raw = worker_generate_one(schedule_row, llama_instance=llm)
            _append_jsonl(raw_path, raw)
            receipts.append(checkpoint_raw_row(checkpoint_path, manifest_hash, raw))
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
        llm = None
        gc.collect()
    rows = load_checkpoint(checkpoint_path, manifest_hash)
    template = _template_text(metadata)
    return {
        "model_id": payload["model_id"],
        "metadata_hash": sha256_text(canonical_json(metadata)),
        "chat_template_present": bool(template),
        "chat_template_hash": sha256_text(template) if template else None,
        "row_count": len(rows),
        "checkpoint_receipts": receipts,
        "model_close_called": True,
        "terminal_state": "complete",
    }


def parse_entrance(raw_text: str) -> JsonDict | None:
    """Parse only the declared compact first-branch JSON schema."""

    text = raw_text.strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"operand_pair", "operator"}:
        return None
    pair = value.get("operand_pair")
    operator = value.get("operator")
    if (
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(type(item) is int for item in pair)
        or operator not in {"+", "-", "*", "/"}
    ):
        return None
    return {"operand_pair": sorted(pair), "operator": operator}


def label_and_project_rows(
    raw_rows: Sequence[Mapping[str, Any]], entrance_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Parse, then exact-label copies without mutating raw rows."""

    exact = {
        (
            str(row.get("unit_id")),
            tuple(sorted(int(value) for value in row.get("operand_pair", []))),
            str(row.get("operator")),
        ): row
        for row in entrance_rows
    }
    raw_output_rows = [deepcopy(dict(row)) for row in raw_rows]
    parse_rows = []
    exact_rows = []
    for row in raw_rows:
        identity = {
            "raw_key": row.get("raw_key"),
            "raw_output_hash": row.get("raw_output_hash"),
            "model_id": row.get("model_id"),
            "unit_id": row.get("unit_id"),
            "seed": row.get("seed"),
        }
        parsed = parse_entrance(str(row.get("raw_text", "")))
        parse_rows.append({**identity, "parsed_entrance": parsed, "parse_failure": parsed is None})
        exact_row = None
        if parsed is not None:
            exact_row = exact.get(
                (
                    str(row.get("unit_id")),
                    tuple(parsed["operand_pair"]),
                    str(parsed["operator"]),
                )
            )
        exact_rows.append(
            {
                **identity,
                "entrance_id": exact_row.get("entrance_id") if exact_row else None,
                "legal": exact_row is not None,
                "reachable": exact_row.get("reachable") is True if exact_row else False,
                "label_source": "experiment_7064_exhaustive_enumerator",
            }
        )
    return {
        "raw_output_rows": raw_output_rows,
        "parse_rows": parse_rows,
        "exact_label_rows": exact_rows,
    }


def _row_rates(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute family-local transport rates from immutable rows."""

    parse_map = {
        str(row.get("raw_key")): parse_entrance(str(row.get("raw_text", ""))) is not None
        for row in raw_rows
    }
    counts: dict[str, JsonDict] = {}
    for model_id in REQUIRED_MODEL_IDS:
        rows = [row for row in raw_rows if row.get("model_id") == model_id]
        total = len(rows)
        counts[model_id] = {
            "row_count": total,
            "empty_count": sum(not str(row.get("raw_text", "")).strip() for row in rows),
            "zero_token_count": sum(int(row.get("completion_tokens", 0) or 0) == 0 for row in rows),
            "parseable_count": sum(parse_map.get(str(row.get("raw_key")), False) for row in rows),
            "leaked_control_token_count": sum(
                CONTROL_TOKEN_RE.search(str(row.get("raw_text", ""))) is not None for row in rows
            ),
            "length_limited_count": sum(row.get("finish_reason") == "length" for row in rows),
        }
    return counts


def _role_messages_match(row: Mapping[str, Any]) -> bool:
    """Validate the closed role sequence and its optional duplicated prompt."""

    messages = row.get("role_messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return False
    if messages[0] != {"role": "system", "content": SYSTEM_MESSAGE}:
        return False
    if not isinstance(messages[1], dict) or messages[1].get("role") != "user":
        return False
    if not isinstance(messages[1].get("content"), str):
        return False
    if "prompt" in row and messages[1]["content"] != row.get("prompt"):
        return False
    if row.get("role_messages_hash") is not None and row.get("role_messages_hash") != sha256_text(
        canonical_json(messages)
    ):
        return False
    return True


def transport_errors(
    raw_rows: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]
) -> list[str]:
    """Recompute readiness without trusting stored aggregate fields."""

    errors = []
    expected_keys = {
        (model_id, f"unit-{index}")
        for model_id in REQUIRED_MODEL_IDS
        for index in range(UNITS_PER_MODEL)
    }
    observed_keys = {(str(row.get("model_id")), str(row.get("unit_id"))) for row in raw_rows}
    if len(raw_rows) != len(REQUIRED_MODEL_IDS) * UNITS_PER_MODEL or len(observed_keys) != len(
        raw_rows
    ):
        errors.append("raw_row_count_or_duplicate_mismatch")
    if any(
        row.get("terminal_state") != "complete"
        or row.get("raw_persisted_before_parse") is not True
        or row.get("parsed_at_write_time") is not False
        or row.get("labeled_at_write_time") is not False
        or row.get("raw_output_hash") != sha256_text(str(row.get("raw_text", "")))
        or row.get("raw_bytes_hex") != str(row.get("raw_text", "")).encode("utf-8").hex()
        for row in raw_rows
    ):
        errors.append("raw_terminal_or_hash_mismatch")
    if any(row.get("transport_method") != "create_chat_completion" for row in raw_rows):
        errors.append("raw_completion_transport_detected")
    if any(
        row.get("chat_template_present") is not True
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("chat_template_hash", "")))
        or not str(row.get("chat_format", "")).strip()
        for row in raw_rows
    ):
        errors.append("template_receipt_invalid")
    if any(not _role_messages_match(row) for row in raw_rows):
        errors.append("role_message_mismatch")
    if any(row.get("stop_config") != GENERATION_CONFIG["stop"] for row in raw_rows):
        errors.append("stop_config_mismatch")
    if any(not str(row.get("raw_text", "")).strip() for row in raw_rows):
        errors.append("empty_output")
    if any(int(row.get("completion_tokens", 0) or 0) == 0 for row in raw_rows):
        errors.append("zero_token_output")
    if any(CONTROL_TOKEN_RE.search(str(row.get("raw_text", ""))) for row in raw_rows):
        errors.append("leaked_control_token")
    if any(
        row.get("finish_reason") == "length"
        or int(row.get("completion_tokens", 0) or 0) > GENERATION_CONFIG["completion_budget_tokens"]
        for row in raw_rows
    ):
        errors.append("length_limited_output")
    rates = _row_rates(raw_rows)
    if any(
        row["row_count"] != UNITS_PER_MODEL
        or row["parseable_count"] / max(1, row["row_count"]) < 0.95
        for row in rates.values()
    ):
        errors.append("per_model_parseability_below_threshold")

    identities = list(evidence.get("model_identity_rows", []))
    identity_ids = {
        str(row.get("model_id"))
        for row in identities
        if row.get("identity_matches") is True
        and row.get("tokenizer_source") == "embedded_gguf"
        and row.get("chat_template_present") is True
        and row.get("chat_template_hash")
    }
    if identity_ids != set(REQUIRED_MODEL_IDS):
        errors.append("model_identity_incomplete")
    executions = list(evidence.get("model_execution_rows", []))
    execution_ids = {str(row.get("model_id")) for row in executions}
    if execution_ids != set(REQUIRED_MODEL_IDS) or any(
        row.get("terminal_state") != "complete"
        or int(row.get("raw_row_count", 0) or 0) != UNITS_PER_MODEL
        or int(row.get("offloaded_layers", 0) or 0) <= 0
        or row.get("used_both_gpus") is not True
        or row.get("cleanup_passed") is not True
        or int(row.get("model_load_count", 0) or 0) != 1
        for row in executions
    ):
        errors.append("model_execution_incomplete")
    checkpoints = list(evidence.get("checkpoint_rows", []))
    checkpoint_ids = {str(row.get("model_id")) for row in checkpoints}
    if checkpoint_ids != set(REQUIRED_MODEL_IDS) or any(
        int(row.get("row_count", 0) or 0) != UNITS_PER_MODEL
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("sha256", "")))
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("manifest_hash", "")))
        for row in checkpoints
    ):
        errors.append("checkpoint_incomplete")
    leases = list(evidence.get("gpu_lease_rows", []))
    lease_models = {str(row.get("model_id")) for row in leases}
    if lease_models != set(REQUIRED_MODEL_IDS) or any(
        not row.get("lease_id")
        or row.get("owner_preserved") is not True
        or row.get("phase_history") != list(lease_api.COMPLETE_PHASE_SEQUENCE)
        or row.get("released") is not True
        or row.get("lease_lost") is not False
        or row.get("signals_sent")
        for row in leases
    ):
        errors.append("lease_identity_or_release_incomplete")
    releases = list(evidence.get("vram_release_rows", []))
    release_ids = {str(row.get("model_id")) for row in releases}
    if release_ids != set(REQUIRED_MODEL_IDS) or any(
        row.get("passed") is not True for row in releases
    ):
        errors.append("vram_release_incomplete")
    runner = dict(evidence.get("runner_receipt") or {})
    if (
        runner.get("backend") != "llama_cpp.Llama"
        or runner.get("cuda_offload") is not True
        or runner.get("transport_method") != "create_chat_completion"
    ):
        errors.append("runner_receipt_invalid")
    if evidence.get("signals_sent"):
        errors.append("signals_sent_nonempty")
    return list(dict.fromkeys(errors))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact without its self-referential digest."""

    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(stable))


def _metrics(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    rates = _row_rates(raw_rows)
    empty = {}
    zero = {}
    parseable = {}
    leaked = {}
    length = {}
    per_model = []
    for model_id, row in rates.items():
        total = int(row["row_count"])
        empty[model_id] = row["empty_count"] / total if total else 0.0
        zero[model_id] = row["zero_token_count"] / total if total else 0.0
        parseable[model_id] = row["parseable_count"] / total if total else 0.0
        leaked[model_id] = row["leaked_control_token_count"]
        length[model_id] = row["length_limited_count"]
        per_model.append(
            {
                "model_id": model_id,
                **deepcopy(row),
                "empty_output_rate": empty[model_id],
                "zero_token_rate": zero[model_id],
                "parseable_rate": parseable[model_id],
            }
        )
    return {
        "empty_output_rate_by_model": empty,
        "zero_token_rate_by_model": zero,
        "parseable_rate_by_model": parseable,
        "leaked_control_token_count_by_model": leaked,
        "length_limited_count_by_model": length,
        "per_model_rows": per_model,
    }


def _all_models_real(model_specs: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]) -> bool:
    identity_ids = {
        str(row.get("model_id"))
        for row in evidence.get("model_identity_rows", [])
        if row.get("identity_matches") is True
    }
    execution_ids = {
        str(row.get("model_id"))
        for row in evidence.get("model_execution_rows", [])
        if int(row.get("offloaded_layers", 0) or 0) > 0
    }
    return bool(
        [row.get("hf_id") for row in model_specs] == list(REQUIRED_MODEL_IDS)
        and not model_spec_errors(model_specs)
        and identity_ids == set(REQUIRED_MODEL_IDS)
        and execution_ids == set(REQUIRED_MODEL_IDS)
    )


def _per_game_results(
    raw_rows: Sequence[Mapping[str, Any]], exact_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    exact = {str(row.get("raw_key")): row for row in exact_rows}
    return [
        {
            "unit_id": unit_id,
            "row_count": sum(row.get("unit_id") == unit_id for row in raw_rows),
            "parseable_count": sum(
                row.get("unit_id") == unit_id
                and parse_entrance(str(row.get("raw_text", ""))) is not None
                for row in raw_rows
            ),
            "legal_count": sum(
                row.get("unit_id") == unit_id
                and exact.get(str(row.get("raw_key")), {}).get("legal") is True
                for row in raw_rows
            ),
        }
        for unit_id in sorted({str(row.get("unit_id")) for row in raw_rows})
    ]


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]] = (),
    entrance_rows: Sequence[Mapping[str, Any]] = (),
    evidence: Mapping[str, Any] | None = None,
    model_order_rows: Sequence[Mapping[str, Any]] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build one schema-complete blocked, partial, null, or positive artifact."""

    raw = [deepcopy(dict(row)) for row in raw_rows]
    facts = deepcopy(dict(evidence or {}))
    identity_rows = deepcopy(
        list(facts.get("model_identity_rows", preconditions.get("model_identity_rows", [])))
    )
    facts["model_identity_rows"] = identity_rows
    views = label_and_project_rows(raw, entrance_rows)
    metrics = _metrics(raw)
    errors = (
        transport_errors(raw, facts)
        if preconditions.get("all_passed") is True
        else ["preconditions_failed"]
    )
    real = _all_models_real(model_specs, facts)
    ready = bool(preconditions.get("all_passed") is True and not errors and real)
    if preconditions.get("all_passed") is not True:
        verdict_class = "blocked"
        honest_verdict = "blocked: a required precondition failed before model generation"
    elif ready:
        verdict_class = "positive"
        honest_verdict = "positive: all three local GGUF families passed bounded chat transport"
    elif len(raw) == len(REQUIRED_MODEL_IDS) * UNITS_PER_MODEL:
        verdict_class = "null"
        honest_verdict = "null: complete canary evidence did not pass the chat transport gate"
    else:
        verdict_class = "partial"
        honest_verdict = "partial: the launched canary did not acquire all scheduled rows"
    gate_rows = [
        gate_row(
            "transport_error_set",
            [],
            errors if errors != ["preconditions_failed"] else [],
            not errors,
        ),
        gate_row("all_models_real", True, real, real),
        gate_row(
            "chat_transport_ready_score",
            1,
            int(ready),
            ready,
        ),
    ]
    execution_rows = deepcopy(list(facts.get("model_execution_rows", [])))
    checkpoint_rows = deepcopy(list(facts.get("checkpoint_rows", [])))
    lease_rows = deepcopy(list(facts.get("gpu_lease_rows", [])))
    release_rows = deepcopy(list(facts.get("vram_release_rows", [])))
    runner = deepcopy(
        dict(
            facts.get("runner_receipt")
            or {
                "backend": "llama_cpp.Llama",
                "cuda_offload": False,
                "transport_method": "create_chat_completion",
            }
        )
    )
    source_hashes = deepcopy(dict(source_artifact_hashes or {}))
    generation_invoked = bool(raw)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "model_bounded_generation"
        if generation_invoked
        else "blocked_no_run",
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": source_hashes,
        "cited_upstream_artifacts": [
            {
                "path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
                "sha256": preconditions.get("fixture_hash"),
                "fields_imported": [
                    "entrance_fixture_ready_score",
                    "model_visible_rows",
                    "entrance_rows",
                ],
            },
            {
                "path": str(LEASE_AUDIT_PATH.relative_to(REPO_ROOT)),
                "sha256": preconditions.get("lease_audit_hash"),
                "fields_imported": ["gpu_lease_cold_audit_ready_score"],
            },
            {
                "path": str(EXP7080_PATH.relative_to(REPO_ROOT)),
                "sha256": source_hashes.get("experiment_7080_v620_three_family_entrance_bank.json"),
                "fields_imported": ["prior_transport_failure"],
            },
            {
                "path": str(EXP6200_PATH.relative_to(REPO_ROOT)),
                "sha256": source_hashes.get(
                    "experiment_6200_three_family_raw_code_transport_canary.json"
                ),
                "fields_imported": ["prior_raw_transport_failure"],
            },
        ],
        "rows": gate_rows,
        "per_game_results": _per_game_results(raw, views["exact_label_rows"]),
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "model_identity_rows": identity_rows,
        "model_execution_rows": execution_rows,
        "chat_template_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "chat_format": row.get("chat_format"),
                "chat_template_present": row.get("chat_template_present"),
                "chat_template_hash": row.get("chat_template_hash"),
                "transport_method": row.get("transport_method"),
            }
            for row in raw
        ],
        "role_message_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "role_messages": deepcopy(row.get("role_messages")),
                "role_messages_hash": row.get("role_messages_hash"),
            }
            for row in raw
        ],
        "rendered_prompt_hash_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "rendered_prompt_hash": row.get("rendered_prompt_hash"),
                "available": row.get("rendered_prompt_hash_available"),
            }
            for row in raw
        ],
        "stop_config_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "stop": deepcopy(row.get("stop_config")),
            }
            for row in raw
        ],
        "raw_output_rows": views["raw_output_rows"],
        "token_count_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
            }
            for row in raw
        ],
        "finish_reason_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "finish_reason": row.get("finish_reason"),
            }
            for row in raw
        ],
        "parse_rows": views["parse_rows"],
        "exact_label_rows": views["exact_label_rows"],
        "per_model_rows": metrics["per_model_rows"],
        "model_order_rows": [deepcopy(dict(row)) for row in model_order_rows],
        "sampling_config": deepcopy(GENERATION_CONFIG),
        "seed_rows": [
            {"purpose": "generation", "seed": GENERATION_SEED},
            {"purpose": "model_order", "seed": RANDOM_SEED},
        ],
        "checkpoint_rows": checkpoint_rows,
        "runner_receipt": runner,
        "generation_invoked": generation_invoked,
        "total_model_count": len({str(row.get("model_id")) for row in raw}),
        "model_load_count_by_stage": {
            "generation": sum(int(row.get("model_load_count", 0) or 0) for row in execution_rows)
        },
        "per_model_duration_s": {
            str(row.get("model_id")): float(row.get("duration_s", 0.0) or 0.0)
            for row in execution_rows
        },
        "stage_gpu_telemetry_rows": deepcopy(list(facts.get("stage_gpu_telemetry_rows", []))),
        "task_gpu_telemetry_rows": deepcopy(list(facts.get("task_gpu_telemetry_rows", []))),
        "peak_vram_by_device": deepcopy(dict(facts.get("peak_vram_by_device", {}))),
        "gpu_lease_rows": lease_rows,
        "vram_release_rows": release_rows,
        "signals_sent": deepcopy(list(facts.get("signals_sent", []))),
        "empty_output_rate_by_model": metrics["empty_output_rate_by_model"],
        "zero_token_rate_by_model": metrics["zero_token_rate_by_model"],
        "parseable_rate_by_model": metrics["parseable_rate_by_model"],
        "leaked_control_token_count_by_model": metrics["leaked_control_token_count_by_model"],
        "length_limited_count_by_model": metrics["length_limited_count_by_model"],
        "all_models_real": real,
        "chat_transport_ready_score": int(ready),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": (
            gate_summary(preconditions.get("checks", []))
            if preconditions.get("all_passed") is not True
            else gate_summary(gate_rows)
        ),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _raw_projection_rows(raw: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute projections that do not need the private exact table."""

    views = label_and_project_rows(raw, [])
    return {
        "chat_template_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "chat_format": row.get("chat_format"),
                "chat_template_present": row.get("chat_template_present"),
                "chat_template_hash": row.get("chat_template_hash"),
                "transport_method": row.get("transport_method"),
            }
            for row in raw
        ],
        "role_message_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "role_messages": deepcopy(row.get("role_messages")),
                "role_messages_hash": row.get("role_messages_hash"),
            }
            for row in raw
        ],
        "rendered_prompt_hash_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "rendered_prompt_hash": row.get("rendered_prompt_hash"),
                "available": row.get("rendered_prompt_hash_available"),
            }
            for row in raw
        ],
        "stop_config_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "stop": deepcopy(row.get("stop_config")),
            }
            for row in raw
        ],
        "token_count_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
            }
            for row in raw
        ],
        "finish_reason_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "finish_reason": row.get("finish_reason"),
            }
            for row in raw
        ],
        "parse_rows": views["parse_rows"],
    }


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, projections, aggregates, verdict, and checksum."""

    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    if errors:
        return errors
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {})):
        errors.append("field_principles_mismatch")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    score = artifact.get("chat_transport_ready_score")
    if type(score) is not int or score not in (0, 1):
        errors.append("readiness_not_bare_int")
    if artifact.get("MODEL_SPECS") != artifact.get("model_specs"):
        errors.append("model_specs_projection_mismatch")
    raw = list(artifact.get("raw_output_rows", []))
    projected = _raw_projection_rows(raw)
    for name, expected in projected.items():
        if artifact.get(name) != expected:
            errors.append(
                "token_count_projection_mismatch"
                if name == "token_count_rows"
                else f"{name}_projection_mismatch"
            )
    exact_rows = list(artifact.get("exact_label_rows", []))
    if len(exact_rows) != len(raw) or any(
        row.get("raw_key") != exact_rows[index].get("raw_key") or "raw_text" in exact_rows[index]
        for index, row in enumerate(raw)
    ):
        errors.append("exact_label_projection_mismatch")
    facts = {
        "model_identity_rows": list(artifact.get("model_identity_rows", [])),
        "model_execution_rows": list(artifact.get("model_execution_rows", [])),
        "checkpoint_rows": list(artifact.get("checkpoint_rows", [])),
        "gpu_lease_rows": list(artifact.get("gpu_lease_rows", [])),
        "vram_release_rows": list(artifact.get("vram_release_rows", [])),
        "runner_receipt": dict(artifact.get("runner_receipt") or {}),
        "signals_sent": list(artifact.get("signals_sent", [])),
    }
    preflight = dict(artifact.get("preconditions_checked") or {}).get("all_passed") is True
    transport = transport_errors(raw, facts) if preflight else ["preconditions_failed"]
    real = _all_models_real(list(artifact.get("model_specs", [])), facts)
    expected_ready = int(preflight and not transport and real)
    if score != expected_ready:
        errors.append("transport_readiness_mismatch")
    metrics = _metrics(raw)
    for name in (
        "empty_output_rate_by_model",
        "zero_token_rate_by_model",
        "parseable_rate_by_model",
        "leaked_control_token_count_by_model",
        "length_limited_count_by_model",
        "per_model_rows",
    ):
        if artifact.get(name) != metrics[name]:
            errors.append(f"{name}_mismatch")
    if artifact.get("all_models_real") is not real:
        errors.append("all_models_real_mismatch")
    if artifact.get("total_model_count") != len({str(row.get("model_id")) for row in raw}):
        errors.append("total_model_count_mismatch")
    generation_invoked = bool(raw)
    if artifact.get("generation_invoked") is not generation_invoked:
        errors.append("generation_invoked_mismatch")
    expected_class = "model_bounded_generation" if generation_invoked else "blocked_no_run"
    if artifact.get("inference_substrate_class") != expected_class:
        errors.append("inference_substrate_class_mismatch")
    verdict_class = str(artifact.get("verdict_class", ""))
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(f"{verdict_class}:"):
        errors.append("honest_verdict_prefix_mismatch")
    if not preflight:
        summary = dict(artifact.get("gate_check_summary") or {})
        if verdict_class != "blocked" or score != 0:
            errors.append("blocked_verdict_mismatch")
        if not all(key in summary for key in ("failed_check", "expected_value", "observed_value")):
            errors.append("blocked_gate_summary_incomplete")
    elif expected_ready and verdict_class != "positive":
        errors.append("positive_verdict_mismatch")
    elif not expected_ready:
        expected_verdict = "null" if len(raw) == 24 else "partial"
        if verdict_class != expected_verdict:
            errors.append("failed_transport_verdict_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _writable(path: Path) -> bool:
    """Probe a destination with an atomic temporary file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7085-write-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
        return True
    except OSError:
        return False


def _lease_probe(devices: Sequence[Mapping[str, Any]]) -> list[JsonDict]:  # pragma: no cover
    """Read current lease journals without sending signals."""

    rows = []
    for device in devices:
        path = lease_api.journal_path_for(LEASE_RUNTIME_DIR, str(device.get("uuid", "")))
        if not path.exists():
            rows.append({"device_uuid": device.get("uuid"), "classification": "available"})
            continue
        try:
            document = lease_api.read_journal(path)
            owner = dict(document.get("owner") or {})
            live = document.get("released") is not True and lease_api.process_start_matches(
                int(owner.get("pid", -1)), int(owner.get("pid_start_ticks", -1))
            )
            rows.append(
                {
                    "device_uuid": device.get("uuid"),
                    "classification": "live_foreign" if live else "available",
                    "journal_path": str(path),
                    "document": document,
                    "signals_sent": [],
                }
            )
        except Exception as exc:  # noqa: BLE001 - unreadable ownership blocks.
            rows.append(
                {
                    "device_uuid": device.get("uuid"),
                    "classification": "unreadable",
                    "journal_path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "signals_sent": [],
                }
            )
    return rows


def _stop_authority_probe() -> JsonDict:  # pragma: no cover
    """Run stop authority in dry-run mode so it cannot signal or write."""

    command = [sys.executable, str(REPO_ROOT / "scripts/run_stop_authority.py"), "--dry-run"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"passed": False, "observed": f"{type(exc).__name__}: {exc}", "command": command}
    return {
        "passed": result.returncode == 0,
        "observed": {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        "command": command,
    }


def _identity_probe_inline(model: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Load embedded metadata in a disposable process."""

    row = embedded_tokenizer_probe(model)
    metadata = dict(row.get("metadata") or {})
    template = _template_text(metadata)
    expected = str(model["hf_id"]).rsplit("/", 1)[-1].removesuffix("-GGUF").lower()
    normalized = re.sub(r"[^a-z0-9]", "", expected)
    observed = re.sub(r"[^a-z0-9]", "", canonical_json(metadata).lower())
    path_observed = re.sub(r"[^a-z0-9]", "", Path(str(model["model_path"])).name.lower())
    return {
        **row,
        "identity_matches": bool(
            row.get("passed") and normalized in observed and normalized in path_observed
        ),
        "expected_identity": expected,
        "tokenizer_source": "embedded_gguf",
        "chat_template_present": bool(template),
        "chat_template_hash": sha256_text(template) if template else None,
        "detected_chat_format": "chat_template.default" if template else None,
    }


def _identity_probe(
    model: Mapping[str, Any], *, runner: Callable[..., Any] = subprocess.run
) -> JsonDict:  # pragma: no cover
    """Exit the metadata process before model weights load."""

    with tempfile.TemporaryDirectory(prefix="carnot-exp7085-identity-") as directory:
        root = Path(directory)
        payload = root / "model.json"
        output = root / "identity.json"
        write_json_atomic(payload, dict(model))
        command = [
            sys.executable,
            "-m",
            MODULE_NAME,
            "--identity-payload",
            str(payload),
            "--identity-output",
            str(output),
        ]
        try:
            result = runner(
                command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180, check=False
            )
            row = (
                json.loads(output.read_text(encoding="utf-8"))
                if output.is_file()
                else {
                    "model_id": model.get("hf_id"),
                    "identity_matches": False,
                    "tokenizer_source": "embedded_gguf",
                    "chat_template_present": False,
                    "error": "identity_child_output_missing",
                }
            )
            row.update(
                {
                    "probe_process_exit_code": result.returncode,
                    "probe_stdout_hash": sha256_text(str(result.stdout)),
                    "probe_stderr_hash": sha256_text(str(result.stderr)),
                    "isolated_process": True,
                }
            )
            if result.returncode != 0:
                row["identity_matches"] = False
            return row
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "model_id": model.get("hf_id"),
                "identity_matches": False,
                "tokenizer_source": "embedded_gguf",
                "chat_template_present": False,
                "error": f"{type(exc).__name__}: {exc}",
                "probe_process_exit_code": None,
                "isolated_process": True,
            }


def collect_preconditions(
    *,
    lease_audit_path: Path = LEASE_AUDIT_PATH,
    expected_lease_audit_hash: str = PINNED_LEASE_AUDIT_SHA256,
    lease_audit_validator: Callable[
        [Mapping[str, Any]], list[str]
    ] = lease_audit_api.validate_artifact,
    fixture_path: Path,
    expected_fixture_hash: str,
    fixture_validator: Callable[..., bool] = fixture_api.validate_artifact,
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
    llama_probe: Callable[[], JsonDict] = llama_cpp_probe,
    lease_probe: Callable[[Sequence[Mapping[str, Any]]], list[JsonDict]] = _lease_probe,
    stop_authority_probe: Callable[[], JsonDict] = _stop_authority_probe,
    identity_probe: Callable[[Mapping[str, Any]], JsonDict] = _identity_probe,
) -> JsonDict:
    """Measure upstream, cache, template, CUDA, ownership, and path gates."""

    lease_audit: JsonDict = {}
    lease_hash = None
    lease_validation_errors = []
    try:
        lease_audit = json.loads(lease_audit_path.read_text(encoding="utf-8"))
        lease_hash = sha256_file(lease_audit_path)
        lease_validation_errors = lease_audit_validator(lease_audit)
    except Exception as exc:  # noqa: BLE001 - exact gate failure is evidence.
        lease_audit = {"read_error": f"{type(exc).__name__}: {exc}"}
    fixture: JsonDict = {}
    fixture_hash = None
    fixture_valid = False
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        fixture_hash = sha256_file(fixture_path)
        fixture_valid = bool(fixture_validator(fixture, check_files=True))
    except Exception as exc:  # noqa: BLE001 - exact gate failure is evidence.
        fixture = {"read_error": f"{type(exc).__name__}: {exc}"}
    gpu = gpu_probe()
    devices = list(gpu.get("devices", []))
    processes = list(gpu.get("processes", []))
    llama = llama_probe()
    leases = lease_probe(devices)
    stop = stop_authority_probe()
    identities = [identity_probe(row) for row in model_specs]
    spec_errors = model_spec_errors(model_specs)
    file_state = {
        str(row.get("hf_id")): Path(str(row.get("model_path", ""))).is_file() for row in model_specs
    }
    exact_gpus = bool(
        gpu.get("query_ok") is True
        and len(devices) == 2
        and [int(row.get("index", -1)) for row in devices] == [0, 1]
        and all("RTX 3090" in str(row.get("name", "")) for row in devices)
        and all(int(row.get("utilization_gpu_pct", 100)) == 0 for row in devices)
    )
    lease_ok = len(leases) == 2 and all(row.get("classification") == "available" for row in leases)
    checks = [
        gate_row(
            "gpu_lease_cold_audit_source_hash",
            expected_lease_audit_hash,
            lease_hash,
            lease_hash == expected_lease_audit_hash,
        ),
        gate_row(
            "gpu_lease_cold_audit_ready_score",
            1,
            lease_audit.get("gpu_lease_cold_audit_ready_score"),
            lease_audit.get("gpu_lease_cold_audit_ready_score") == 1
            and not lease_validation_errors,
        ),
        gate_row(
            "entrance_fixture_source_hash",
            expected_fixture_hash,
            fixture_hash,
            fixture_hash == expected_fixture_hash,
        ),
        gate_row(
            "entrance_fixture_ready_score",
            1,
            fixture.get("entrance_fixture_ready_score"),
            fixture_valid and fixture.get("entrance_fixture_ready_score") == 1,
        ),
        gate_row("exact_model_specs", [], spec_errors, not spec_errors),
        gate_row(
            "all_three_cached_gguf_files",
            {model_id: True for model_id in REQUIRED_MODEL_IDS},
            file_state,
            len(file_state) == 3 and all(file_state.values()),
        ),
        gate_row(
            "model_identity_rows",
            [],
            model_identity_errors(model_specs, identities),
            not model_identity_errors(model_specs, identities),
        ),
        gate_row("owned_idle_rtx_3090_devices", 2, devices, exact_gpus),
        gate_row("unattributed_gpu_processes", [], processes, not processes),
        gate_row(
            "cuda_llama_cpp",
            {"importable": True, "gpu_offload": True},
            llama,
            llama.get("importable") is True and llama.get("gpu_offload") is True,
        ),
        gate_row("owned_gpu_leases_available", True, leases, lease_ok),
        gate_row("clean_stop_authority", True, stop.get("observed"), stop.get("passed") is True),
        gate_row("result_path_writable", True, str(result_path), _writable(result_path)),
        gate_row(
            "checkpoint_path_writable", True, str(checkpoint_path), _writable(checkpoint_path)
        ),
    ]
    return {
        "all_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "lease_audit": lease_audit,
        "lease_audit_hash": lease_hash,
        "lease_audit_validation_errors": lease_validation_errors,
        "fixture": fixture,
        "fixture_hash": fixture_hash,
        "gpu_topology": gpu,
        "gpu_lease_preflight_rows": leases,
        "model_identity_rows": identities,
        "runner_build_rows": [llama],
        "runner_receipt": {
            **deepcopy(llama),
            "backend": "llama_cpp.Llama",
            "cuda_offload": llama.get("gpu_offload") is True,
            "transport_method": "create_chat_completion",
        },
        "stop_authority": stop,
        "upstream_gate_rows": [deepcopy(row) for row in checks[:4]],
        "signals_sent": [],
    }


def _source_hashes() -> JsonDict:  # pragma: no cover
    """Hash reviewed code, specs, and prior artifacts."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-references.md"),
        Path("ops/known-issues.md"),
        Path("openspec/capabilities/research-harnesses/spec.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("python/carnot/experiment_7085_v621_chat_transport_canary.py"),
        Path("tests/python/test_experiment_7085_v621_chat_transport_canary.py"),
        Path("scripts/experiments/experiment_7085_v621_chat_transport_canary.py"),
        FIXTURE_PATH.relative_to(REPO_ROOT),
        LEASE_AUDIT_PATH.relative_to(REPO_ROOT),
        EXP7080_PATH.relative_to(REPO_ROOT),
        EXP6200_PATH.relative_to(REPO_ROOT),
    )
    files = []
    for relative in paths:
        path = REPO_ROOT / relative
        files.append(
            {
                "path": str(relative),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    by_name = {Path(row["path"]).name: row["sha256"] for row in files}
    return {
        "all_present": all(row["exists"] for row in files),
        "files": files,
        "by_name": by_name,
        "manifest_hash": sha256_text(canonical_json(files)),
    }


def _memory_rows(gpu: Mapping[str, Any]) -> list[JsonDict]:  # pragma: no cover
    return [
        {
            "device_uuid": row.get("uuid"),
            "index": row.get("index"),
            "memory_used_mb": int(row.get("memory_used_mb", 0) or 0),
        }
        for row in gpu.get("devices", [])
    ]


def _wait_vram_release(
    baseline_rows: Sequence[Mapping[str, Any]], model_id: str
) -> JsonDict:  # pragma: no cover
    deadline = time.monotonic() + VRAM_RELEASE_TIMEOUT_S
    while True:
        after_gpu = gpu_inventory()
        after_rows = _memory_rows(after_gpu)
        row = build_vram_release_row(
            model_id=model_id, baseline_rows=baseline_rows, after_rows=after_rows
        )
        row["after_rows"] = after_rows
        if row["passed"] is True or time.monotonic() >= deadline:
            return row
        time.sleep(1)


def _choose_free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_owner_pids(port: int) -> list[int]:  # pragma: no cover
    result = subprocess.run(
        ["ss", "-ltnp", f"sport = :{int(port)}"], capture_output=True, text=True, check=False
    )
    return sorted({int(value) for value in re.findall(r"pid=(\d+)", result.stdout)})


def latch_ready_owned(
    ready_owned: bool,
    ready: Mapping[str, Any],
    expected_pid: int,
    expected_start_ticks: int | None,
    expected_port: int,
    port_owner_pids: Sequence[int],
) -> bool:
    """Retain a proven worker-listener ownership receipt through shutdown."""

    return bool(
        ready_owned
        or (
            ready.get("pid") == expected_pid
            and ready.get("pid_start_ticks") == expected_start_ticks
            and ready.get("port") == expected_port
            and list(port_owner_pids) == [expected_pid]
        )
    )


def _port_is_free(port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def _acquire_leases(
    model: Mapping[str, Any], devices: Sequence[Mapping[str, Any]]
) -> tuple[list[Any], list[JsonDict]]:  # pragma: no cover
    leases = []
    rows = []
    try:
        for device in devices:
            lease = lease_api.GpuLease.acquire(
                runtime_dir=LEASE_RUNTIME_DIR,
                task_id=EXPERIMENT_ID,
                device_uuid=str(device["uuid"]),
                expected_model=str(model["model_path"]),
                vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
                ttl_s=MODEL_TIMEOUT_S + 300,
            )
            lease.transition("admitted")
            lease.transition("loading")
            leases.append(lease)
            rows.append(
                {
                    "model_id": model["hf_id"],
                    "device_uuid": device["uuid"],
                    "lease_id": lease.lease_id,
                    "owner_pid": lease.pid,
                    "owner_pid_start_ticks": lease.pid_start_ticks,
                    "journal_after_acquisition": deepcopy(lease.document),
                    "journal_after_release": None,
                    "release_receipt": None,
                    "signals_sent": [],
                }
            )
    except Exception:
        for lease in leases:
            try:
                lease.transition("terminal_blocked")
                lease.release()
            except Exception:  # noqa: BLE001 - close only leases already owned.
                lease.close()
        raise
    return leases, rows


def _release_leases(
    leases: Sequence[Any],
    rows: list[JsonDict],
    *,
    resident: bool,
    complete: bool,
    release: Mapping[str, Any],
    exit_code: int,
) -> None:  # pragma: no cover
    after = {
        str(row["device_uuid"]): int(row["memory_used_mb"]) for row in release.get("after_rows", [])
    }
    for lease, row in zip(leases, rows, strict=True):
        try:
            if resident:
                lease.transition("unloading")
                lease.transition(
                    "validating",
                    vram_mb=after.get(lease.device_uuid, 0),
                    exit_code=exit_code,
                    unload_observed=release.get("passed") is True,
                )
                lease.transition("terminal_complete" if complete else "terminal_blocked")
            else:
                lease.transition("terminal_blocked")
            row["release_receipt"] = lease.release()
            row["journal_after_release"] = deepcopy(lease.document)
        except Exception as exc:  # noqa: BLE001 - release failure remains evidence.
            lease.close()
            row["release_error"] = f"{type(exc).__name__}: {exc}"


def _normalize_lease_rows(phase_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce live lease journals to the ownership facts used by readiness."""

    rows = []
    for phase in phase_rows:
        for source in phase.get("gpu_lease_rows", []):
            acquired = dict(source.get("journal_after_acquisition") or {})
            released = dict(source.get("journal_after_release") or {})
            receipt = dict(source.get("release_receipt") or {})
            acquired_owner = dict(acquired.get("owner") or {})
            released_owner = dict(released.get("owner") or {})
            lease_id = source.get("lease_id") or acquired.get("lease_id")
            rows.append(
                {
                    "model_id": phase.get("model_id"),
                    "device_uuid": source.get("device_uuid"),
                    "lease_id": lease_id,
                    "owner_preserved": bool(
                        lease_id
                        and receipt.get("lease_id") == lease_id
                        and released.get("lease_id") == lease_id
                        and acquired_owner == released_owner
                    ),
                    "phase_history": [
                        str(item.get("phase")) for item in released.get("phase_history", [])
                    ],
                    "released": receipt.get("released") is True
                    and released.get("released") is True,
                    "lease_lost": bool(
                        source.get("release_error")
                        or not lease_id
                        or receipt.get("lease_id") != lease_id
                        or released.get("lease_id") != lease_id
                        or acquired_owner != released_owner
                    ),
                    "journal_checksum": released.get("checksum"),
                    "signals_sent": deepcopy(list(source.get("signals_sent", []))),
                }
            )
    return rows


def run_model_phase(
    *,
    model: Mapping[str, Any],
    schedule_rows: Sequence[Mapping[str, Any]],
    devices: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    checkpoint_dir: Path,
) -> JsonDict:  # pragma: no cover
    """Run one owned CUDA worker and prove its release."""

    phase_started = time.perf_counter()
    baseline = gpu_inventory()
    if baseline.get("processes"):
        return {
            "model_id": model["hf_id"],
            "terminal_state": "blocked",
            "error": "unattributed_gpu_process",
            "cleanup": {"model_id": model["hf_id"], "passed": False, "signals_sent": []},
        }
    baseline_rows = _memory_rows(baseline)
    leases, lease_rows = _acquire_leases(model, devices)
    slug = re.sub(r"[^a-z0-9]+", "-", str(model["hf_id"]).lower()).strip("-")
    phase_dir = raw_dir / slug
    phase_dir.mkdir(parents=True, exist_ok=True)
    raw_path = phase_dir / "raw.jsonl"
    checkpoint_path = checkpoint_dir / f"{slug}.json"
    payload_path = phase_dir / "payload.json"
    output_path = phase_dir / "worker-output.json"
    ready_path = phase_dir / "ready.json"
    stdout_path = phase_dir / "stdout.log"
    stderr_path = phase_dir / "stderr.log"
    manifest_hash = sha256_text(
        canonical_json({"model": model, "rows": list(schedule_rows), "config": GENERATION_CONFIG})
    )
    write_json_atomic(
        payload_path,
        {
            "model_id": model["hf_id"],
            "model_path": model["model_path"],
            "schedule_rows": list(schedule_rows),
            "raw_path": str(raw_path),
            "checkpoint_path": str(checkpoint_path),
            "manifest_hash": manifest_hash,
        },
    )
    port = _choose_free_port()
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--worker-payload",
        str(payload_path),
        "--worker-output",
        str(output_path),
        "--worker-ready",
        str(ready_path),
        "--worker-port",
        str(port),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    samples = []
    ready_owned = False
    resident = False
    signals: list[str] = []
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        start_ticks = lease_api.proc_start_ticks(process.pid)
        deadline = time.monotonic() + MODEL_TIMEOUT_S
        while process.poll() is None and time.monotonic() < deadline:
            sample = gpu_inventory()
            sample["monotonic_ns"] = time.monotonic_ns()
            samples.append(sample)
            if ready_path.is_file():
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
                ready_owned = latch_ready_owned(
                    ready_owned,
                    ready,
                    process.pid,
                    start_ticks,
                    port,
                    _port_owner_pids(port),
                )
            used = {
                str(row.get("gpu_uuid"))
                for row in sample.get("processes", [])
                if row.get("pid") == process.pid
            }
            expected = {str(row["uuid"]) for row in devices}
            if used == expected and not resident:
                memory = {str(row["uuid"]): int(row["memory_used_mb"]) for row in sample["devices"]}
                for lease in leases:
                    lease.transition("resident", vram_mb=memory.get(lease.device_uuid, 0))
                    lease.transition("inferencing")
                resident = True
            time.sleep(0.5)
        if process.poll() is None:
            if lease_api.proc_start_ticks(process.pid) == start_ticks:
                os.killpg(process.pid, 15)
                signals.append("SIGTERM")
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if lease_api.proc_start_ticks(process.pid) == start_ticks:
                    os.killpg(process.pid, 9)
                    signals.append("SIGKILL")
        process.wait(timeout=60)
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    offload = parse_offloaded_layers(stderr_text)
    release = _wait_vram_release(baseline_rows, str(model["hf_id"]))
    rows = load_checkpoint(checkpoint_path, manifest_hash) if checkpoint_path.is_file() else []
    expected_keys = {str(row["raw_key"]) for row in schedule_rows}
    complete = bool(
        process.returncode == 0
        and ready_owned
        and resident
        and offload["offloaded"] > 0
        and {str(row["raw_key"]) for row in rows} == expected_keys
        and all(row.get("terminal_state") == "complete" for row in rows)
        and release.get("passed") is True
        and not signals
    )
    _release_leases(
        leases,
        lease_rows,
        resident=resident,
        complete=complete,
        release=release,
        exit_code=int(process.returncode or 0),
    )
    leases_released = all(
        dict(row.get("journal_after_release") or {}).get("released") is True for row in lease_rows
    )
    process_absent = lease_api.proc_start_ticks(process.pid) != start_ticks
    port_free = _port_is_free(port)
    cleanup = {
        "model_id": model["hf_id"],
        "process_owned": ready_owned,
        "owned_process_absent": process_absent,
        "port_release_confirmed": port_free,
        "gpu_leases_released": leases_released,
        "vram_release_passed": release.get("passed") is True,
        "signals_sent": signals,
        "signaled_pid_owned": ready_owned,
        "passed": bool(
            ready_owned
            and process_absent
            and port_free
            and leases_released
            and release.get("passed") is True
            and not signals
        ),
    }
    owned_samples = [
        {"monotonic_ns": sample["monotonic_ns"], **deepcopy(dict(proc))}
        for sample in samples
        for proc in sample.get("processes", [])
        if proc.get("pid") == process.pid
    ]
    return {
        "model_id": model["hf_id"],
        "terminal_state": "complete" if complete else "failed",
        "raw_rows": rows,
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
        "manifest_hash": manifest_hash,
        "worker_output": json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.is_file()
        else {},
        "offloaded_layers": offload["offloaded"],
        "total_layers": offload["total"],
        "used_both_gpus": len({row.get("gpu_uuid") for row in owned_samples}) == 2,
        "gpu_sample_rows": owned_samples,
        "task_gpu_samples": samples,
        "gpu_lease_rows": lease_rows,
        "cleanup": cleanup,
        "vram_release": release,
        "backend_stderr_hash": sha256_text(stderr_text),
        "duration_s": time.perf_counter() - phase_started,
        "model_load_count": 1,
    }


def _worker_main(
    payload_path: Path, output_path: Path, ready_path: Path, port: int
) -> int:  # pragma: no cover
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", int(port)))
        listener.listen(1)
        write_json_atomic(
            ready_path,
            {
                "pid": os.getpid(),
                "pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
                "port": port,
            },
        )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result = worker_run_schedule(payload)
        write_json_atomic(output_path, result)
        return 0
    except Exception as exc:  # noqa: BLE001 - parent retains exact worker failure.
        write_json_atomic(
            output_path,
            {
                "terminal_state": "failed",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "exception_traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        listener.close()


def _phase_evidence(
    phase_rows: Sequence[Mapping[str, Any]], preconditions: Mapping[str, Any]
) -> JsonDict:
    """Project live phase receipts into the terminal evidence schema."""

    executions = [
        {
            "model_id": row.get("model_id"),
            "terminal_state": row.get("terminal_state"),
            "raw_row_count": len(row.get("raw_rows", [])),
            "offloaded_layers": row.get("offloaded_layers"),
            "total_layers": row.get("total_layers"),
            "used_both_gpus": row.get("used_both_gpus"),
            "cleanup_passed": dict(row.get("cleanup") or {}).get("passed"),
            "model_load_count": row.get("model_load_count"),
            "duration_s": row.get("duration_s"),
            "backend_stderr_hash": row.get("backend_stderr_hash"),
        }
        for row in phase_rows
    ]
    checkpoints = [
        {
            "model_id": row.get("model_id"),
            "path": row.get("checkpoint_path"),
            "sha256": row.get("checkpoint_sha256"),
            "manifest_hash": row.get("manifest_hash"),
            "row_count": len(row.get("raw_rows", [])),
        }
        for row in phase_rows
    ]
    releases = [
        {"model_id": row.get("model_id"), **deepcopy(dict(row.get("vram_release") or {}))}
        for row in phase_rows
    ]
    stage_samples = [
        {"model_id": row.get("model_id"), **deepcopy(dict(sample))}
        for row in phase_rows
        for sample in row.get("gpu_sample_rows", [])
    ]
    task_samples = [
        {"model_id": row.get("model_id"), **deepcopy(dict(sample))}
        for row in phase_rows
        for sample in row.get("task_gpu_samples", [])
    ]
    peaks: dict[str, int] = {}
    for sample in task_samples:
        for device in sample.get("devices", []):
            uuid = str(device.get("uuid"))
            peaks[uuid] = max(peaks.get(uuid, 0), int(device.get("memory_used_mb", 0) or 0))
    signals = [
        signal
        for row in phase_rows
        for signal in dict(row.get("cleanup") or {}).get("signals_sent", [])
    ]
    return {
        "model_identity_rows": deepcopy(list(preconditions.get("model_identity_rows", []))),
        "model_execution_rows": executions,
        "checkpoint_rows": checkpoints,
        "gpu_lease_rows": _normalize_lease_rows(phase_rows),
        "vram_release_rows": releases,
        "runner_receipt": deepcopy(dict(preconditions.get("runner_receipt") or {})),
        "signals_sent": signals,
        "stage_gpu_telemetry_rows": stage_samples,
        "task_gpu_telemetry_rows": task_samples,
        "peak_vram_by_device": peaks,
    }


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    fixture_path: Path = FIXTURE_PATH,
    raw_dir: Path = RAW_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:  # pragma: no cover
    """Preflight, run three bounded workers, label, validate, and write."""

    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    preconditions = collect_preconditions(
        fixture_path=fixture_path,
        expected_fixture_hash=PINNED_FIXTURE_SHA256,
        model_specs=specs,
        result_path=result_path,
        checkpoint_path=checkpoint_dir / "write-probe.json",
    )
    source_hashes = _source_hashes()
    source_by_name = dict(source_hashes.get("by_name") or {})
    if preconditions.get("all_passed") is not True:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            preconditions=preconditions,
            source_artifact_hashes=source_by_name,
        )
        write_json_atomic(result_path, artifact)
        return artifact
    fixture = dict(preconditions["fixture"])
    held_groups = set(fixture["split_manifest"]["held_source_group_ids"])
    private_units = [row for row in fixture["unit_rows"] if row["source_group_id"] in held_groups]
    selected_private = select_representative_units(private_units)
    visible_by_id = {str(row["unit_id"]): row for row in fixture["model_visible_rows"]}
    selected = [
        {
            **deepcopy(dict(visible_by_id[str(row["unit_id"])])),
            "source_group_id": row["source_group_id"],
        }
        for row in selected_private
    ]
    schedule = build_schedule(specs, selected)
    schedule_failures = schedule_errors(schedule, [str(row["unit_id"]) for row in selected])
    if schedule_failures:
        raise RuntimeError(f"schedule_invalid:{schedule_failures}")
    order_rows = model_order_rows()
    devices = preconditions["gpu_topology"]["devices"]
    phases = []
    for order in order_rows:
        model = next(row for row in specs if row["hf_id"] == order["model_id"])
        phase = run_model_phase(
            model=model,
            schedule_rows=[row for row in schedule if row["model_id"] == model["hf_id"]],
            devices=devices,
            raw_dir=raw_dir,
            checkpoint_dir=checkpoint_dir,
        )
        phases.append(phase)
        if phase.get("terminal_state") != "complete":
            break
    raw = [row for phase in phases for row in phase.get("raw_rows", [])]
    raw.sort(key=lambda row: str(row.get("raw_key")))
    evidence = _phase_evidence(phases, preconditions)
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions=preconditions,
        raw_rows=raw,
        entrance_rows=fixture["entrance_rows"],
        evidence=evidence,
        model_order_rows=order_rows,
        source_artifact_hashes=source_by_name,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--fixture-path", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-ready", type=Path)
    parser.add_argument("--worker-port", type=int)
    parser.add_argument("--identity-payload", type=Path)
    parser.add_argument("--identity-output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.identity_payload:
        if not args.identity_output:
            parser.error("identity mode requires an output path")
        write_json_atomic(
            args.identity_output,
            _identity_probe_inline(json.loads(args.identity_payload.read_text(encoding="utf-8"))),
        )
        return 0
    if args.worker_payload:
        if not args.worker_output or not args.worker_ready or args.worker_port is None:
            parser.error("worker mode requires output, ready, and port")
        return _worker_main(
            args.worker_payload, args.worker_output, args.worker_ready, args.worker_port
        )
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        fixture_path=args.fixture_path,
        raw_dir=args.raw_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "chat_transport_ready_score": artifact["chat_transport_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
