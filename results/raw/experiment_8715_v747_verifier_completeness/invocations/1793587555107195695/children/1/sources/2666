"""Acquire a matched three-family bank of first arithmetic branches.

Spec refs: REQ-INFRA-7065, REQ-VERIFY-7065, and their scenarios.

The model worker writes raw bytes and token scores before any parser runs. The
controller opens the Exp7064 exact labels only after the proposal workers exit.
No code in this module fits or selects an energy model.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
MODULE_NAME = "carnot.experiment_7065_v619_three_family_entrance_bank"
EXPERIMENT_ID = "experiment_7065_v619_three_family_entrance_bank"
SCHEMA = "carnot.experiment_7065.v619_three_family_entrance_bank.v1"
RUN_DATE = "20260906"
RANDOM_SEED = 7_065_202_609_06
INFERENCE_SUBSTRATE = "live_llm_inference"
RESULT_PATH = REPO_ROOT / "results/experiment_7065_v619_three_family_entrance_bank.json"
FIXTURE_PATH = REPO_ROOT / "results/experiment_7064_v619_exact_entrance_fixture.json"
PINNED_FIXTURE_SHA256 = "sha256:6b62768e3387d40eebf462c199aab6a440321aa4a1549ff7d54faaba312f2277"
RAW_DIR = REPO_ROOT / "results/raw/experiment_7065_v619_three_family_entrance_bank"
CHECKPOINT_DIR = REPO_ROOT / "results/checkpoints/experiment_7065_v619_three_family_entrance_bank"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
PREFERRED_QUANT = "Q4_K_M"
PROPOSAL_SEEDS = (7_065_001, 7_065_002, 7_065_003, 7_065_004)
FORCED_PREFIX_SEED = 7_065_101
FORCED_UNIT_COUNT = 24
MODEL_TIMEOUT_S = 7_200.0
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
    "temperature": 0.7,
    "top_p": 0.9,
    "completion_budget_tokens": 64,
    "stop": ["\n\n"],
    "visible_devices": [0, 1],
    "tokenizer_source": "embedded_gguf",
}

PROMPT_TEMPLATE = (
    "Use the available positive integers exactly as a multiset. You may combine two values "
    "with +, -, *, or exact integer /. Subtraction must stay positive. The target is {target}. "
    "Available integers: {numbers}. Propose only the first arithmetic branch. Return only one "
    'JSON object on one line: {{"operand_pair":[smaller,larger],"operator":"+|-|*|/"}}. '
    "Do not solve the rest and do not add prose."
)
FORCED_TEMPLATE = (
    "{proposal_prompt}\nThe first branch is forced to {prefix_json}. Continue from its result "
    "under the same arithmetic rules. Return only a JSON array of the remaining operations. "
    "Each operation must have left, right, operator, and result. Do not repeat the forced branch."
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "upstream_gate_rows",
    "rows",
    "per_game_results",
    "MODEL_SPECS",
    "model_specs",
    "models_used",
    "selected_model_specs",
    "model_identity_rows",
    "model_file_hash_rows",
    "runner_build_rows",
    "gpu_lease_rows",
    "port_lease_rows",
    "gpu_sample_rows",
    "cuda_layer_offload_confirmed",
    "prompt_template",
    "prompt_hash",
    "matched_budget_contract",
    "proposal_rows",
    "forced_prefix_rows",
    "raw_output_manifest",
    "raw_output_hashes",
    "checkpoint_rows",
    "cleanup_rows",
    "model_family_count",
    "entrance_proposal_bank_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for every required field makes missing evidence visible.",
    "preconditions_checked": "Measured resource gates prevent fabricated model evidence.",
    "inference_substrate": "The declared substrate distinguishes live inference from replay.",
    "duration_s": "Wall time exposes truncated or synthetic acquisition.",
    "source_artifact_hashes": "Source hashes bind output to the reviewed code and fixture.",
    "cited_upstream_artifacts": "Explicit citations identify the frozen input artifact.",
    "upstream_gate_rows": "Bare upstream gates prevent inference on an unready fixture.",
    "rows": "Recomputable gate rows separate completeness from prose.",
    "per_game_results": "Unit summaries expose support without hiding failed attempts.",
    "MODEL_SPECS": "The exact ordered roster prevents a small-model substitution.",
    "model_specs": "A lowercase mirror supports consumers with the newer field name.",
    "models_used": "Executed family IDs make three-family coverage falsifiable.",
    "selected_model_specs": "Execution order proves the model shuffle was preregistered.",
    "model_identity_rows": "Embedded metadata binds outputs to the declared GGUF family.",
    "model_file_hash_rows": "Byte hashes prevent silent weight or quantization drift.",
    "runner_build_rows": "Runner version and CUDA support identify the inference engine.",
    "gpu_lease_rows": "Owner-bound leases prove exclusive device authority.",
    "port_lease_rows": "A private readiness port binds each worker to its controller.",
    "gpu_sample_rows": "PID-linked samples show that inference reached both GPUs.",
    "cuda_layer_offload_confirmed": "Layer evidence rules out a CPU headline fallback.",
    "prompt_template": "Frozen prompt text prevents model-specific instruction changes.",
    "prompt_hash": "One prompt hash detects prompt-byte drift.",
    "matched_budget_contract": "Matched sampling and token limits isolate model family.",
    "proposal_rows": "Post-generation labels measure legal and reachable first branches.",
    "forced_prefix_rows": "Forced continuations measure support beyond initial selection.",
    "raw_output_manifest": "Shard paths and hashes preserve output before parsing.",
    "raw_output_hashes": "Per-attempt hashes detect changed or reused raw output.",
    "checkpoint_rows": "Resume receipts preserve completed attempts without duplication.",
    "cleanup_rows": "Owned teardown evidence prevents cross-family contamination.",
    "model_family_count": "A bare count makes the three-family minimum mechanical.",
    "entrance_proposal_bank_complete_score": "One means complete data, not good proposals.",
    "random_seed": "A fixed controller seed reproduces model and arm order.",
    "reproducibility_checksum": "A terminal content hash detects artifact mutation.",
    "gate_check_summary": "Exact expected and observed values make failures actionable.",
    "verifier_is_oracle": "False keeps exact labels from becoming a model-quality claim.",
    "verdict_class": "A closed class gives automation an unambiguous terminal state.",
    "honest_verdict": "A class-consistent prefix reports completion without quality inflation.",
}


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key and byte ordering."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash text after its exact UTF-8 encoding."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the mandated dense pair first, then add the third family."""

    pair = cached_pair_func(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    pair_paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows: list[JsonDict] = []
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
                    else "resolve_cached_gguf exact family extension"
                ),
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject missing, reordered, remote, legacy, or non-primary model paths."""

    errors: list[str] = []
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
    """Require one successful embedded-tokenizer identity row per model."""

    by_id = {str(row.get("model_id")): row for row in rows}
    errors: list[str] = []
    for spec in specs:
        model_id = str(spec.get("hf_id"))
        row = by_id.get(model_id, {})
        if row.get("identity_matches") is not True:
            errors.append(f"model_identity_mismatch:{model_id}")
        elif row.get("tokenizer_source", "embedded_gguf") != "embedded_gguf":
            errors.append(f"model_tokenizer_mismatch:{model_id}")
    return errors


def build_proposal_prompt(unit: Mapping[str, Any]) -> str:
    """Render one label-free first-branch prompt from model-visible fields."""

    return PROMPT_TEMPLATE.format(
        target=int(unit["target"]),
        numbers=canonical_json([int(value) for value in unit["numbers"]]),
    )


def build_proposal_schedule(
    specs: Sequence[Mapping[str, Any]],
    visible_units: Sequence[Mapping[str, Any]],
    held_unit_ids: Sequence[str],
    *,
    seed: int = RANDOM_SEED,
) -> list[JsonDict]:
    """Build the exact cross-product, then shuffle only execution order."""

    visible = {str(row["unit_id"]): row for row in visible_units}
    rows = []
    for model in specs:
        for unit_id in held_unit_ids:
            prompt = build_proposal_prompt(visible[unit_id])
            for proposal_seed in PROPOSAL_SEEDS:
                rows.append(
                    {
                        "raw_key": f"{model['hf_id']}|proposal|{unit_id}|{proposal_seed}",
                        "model_id": str(model["hf_id"]),
                        "model_path": str(model["model_path"]),
                        "unit_id": unit_id,
                        "seed": proposal_seed,
                        "arm": "proposal",
                        "prompt": prompt,
                        "prompt_hash": sha256_text(prompt),
                        "generation_config": deepcopy(GENERATION_CONFIG),
                    }
                )
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["execution_index"] = index
    return rows


def matched_schedule_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check hashes and cross-model prompt and budget identity."""

    errors: list[str] = []
    if any(row.get("prompt_hash") != sha256_text(str(row.get("prompt", ""))) for row in rows):
        errors.append("prompt_hash_mismatch")
    units: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        units.setdefault(str(row.get("unit_id")), []).append(row)
    for unit_id, unit_rows in units.items():
        if len({str(row.get("prompt")) for row in unit_rows}) != 1:
            errors.append(f"cross_model_prompt_mismatch:{unit_id}")
        if len({canonical_json(row.get("generation_config")) for row in unit_rows}) != 1:
            errors.append(f"cross_model_budget_mismatch:{unit_id}")
        by_model: dict[str, set[int]] = {}
        for row in unit_rows:
            by_model.setdefault(str(row.get("model_id")), set()).add(int(row.get("seed", -1)))
        if any(seeds != set(PROPOSAL_SEEDS) for seeds in by_model.values()):
            errors.append(f"proposal_seed_mismatch:{unit_id}")
    return list(dict.fromkeys(errors))


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one synced line so a crash cannot expose a derived-only row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def persist_raw_then_label(
    path: Path,
    row: Mapping[str, Any],
    labeler: Callable[[JsonDict], JsonDict],
) -> JsonDict:
    """Persist raw evidence before calling an injected parser or labeler."""

    raw = deepcopy(dict(row))
    _append_jsonl(path, raw)
    return labeler(deepcopy(raw))


def checkpoint_raw_row(path: Path, manifest_hash: str, row: Mapping[str, Any]) -> JsonDict:
    """Add one immutable raw row to an atomic manifest-bound checkpoint."""

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


def load_checkpoint(path: Path, manifest_hash: str) -> list[JsonDict]:
    """Load a checkpoint only when its manifest and unique keys still match."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("manifest_hash") != manifest_hash:
        raise ValueError("checkpoint_manifest_mismatch")
    rows = [deepcopy(dict(row)) for row in document.get("rows", [])]
    keys = [str(row.get("raw_key")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("checkpoint_duplicate_raw_key")
    return rows


def _extract_token_scores(choice: Mapping[str, Any]) -> list[JsonDict]:
    """Normalize legacy llama.cpp logprobs without storing vocabulary vectors."""

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, Mapping):
        return []
    tokens = list(logprobs.get("tokens") or [])
    values = list(logprobs.get("token_logprobs") or [])
    offsets = list(logprobs.get("text_offset") or [])
    tops = list(logprobs.get("top_logprobs") or [])
    return [
        {
            "token": str(token),
            "logprob": values[index] if index < len(values) else None,
            "text_offset": offsets[index] if index < len(offsets) else None,
            "top_logprobs_hash": sha256_text(canonical_json(tops[index]))
            if index < len(tops)
            else None,
        }
        for index, token in enumerate(tokens)
    ]


def worker_generate_one(
    schedule_row: Mapping[str, Any],
    *,
    llama_factory: Callable[..., Any] | None = None,
    llama_instance: Any = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Generate one raw row and keep failures in the same evidence shape."""

    owns_llama = llama_instance is None
    llm = llama_instance
    started_ns = clock()
    close_called = False
    try:
        config = dict(schedule_row.get("generation_config", GENERATION_CONFIG))
        if llm is None:
            if llama_factory is None:  # pragma: no cover - exercised by the live child.
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
                logits_all=True,
                seed=int(schedule_row["seed"]),
                verbose=True,
            )
        max_tokens = int(config["completion_budget_tokens"])
        prefix_token_count = 0
        if schedule_row.get("arm") == "forced_prefix":
            prefix = str(schedule_row.get("prefix_json", ""))
            prefix_token_count = len(llm.tokenize(prefix.encode("utf-8"), add_bos=False))
            max_tokens = max(1, max_tokens - prefix_token_count)
        response = llm.create_completion(
            str(schedule_row["prompt"]),
            max_tokens=max_tokens,
            temperature=float(config["temperature"]),
            top_p=float(config["top_p"]),
            seed=int(schedule_row["seed"]),
            stop=list(config["stop"]),
            logprobs=5,
        )
        ended_ns = clock()
        choice = (response.get("choices") or [{}])[0]
        usage = dict(response.get("usage") or {})
        text = str(choice.get("text") or "")
        row = {
            **deepcopy(dict(schedule_row)),
            "raw_text": text,
            "raw_output_hash": sha256_text(text),
            "token_scores": _extract_token_scores(choice),
            "timings": {
                "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
                "backend": deepcopy(dict(response.get("timings") or {})),
            },
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "prefix_token_count": prefix_token_count,
            "effective_completion_budget_tokens": max_tokens,
            "finish_reason": str(choice.get("finish_reason") or "unknown"),
            "terminal_state": "complete",
            "exception_type": None,
            "exception_message": None,
            "raw_persisted_before_parse": True,
            "parsed_at_write_time": False,
            "labeled_at_write_time": False,
        }
    except Exception as exc:  # noqa: BLE001 - backend failure is raw evidence.
        ended_ns = clock()
        row = {
            **deepcopy(dict(schedule_row)),
            "raw_text": "",
            "raw_output_hash": sha256_text(""),
            "token_scores": [],
            "timings": {
                "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
                "backend": {},
            },
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prefix_token_count": 0,
            "effective_completion_budget_tokens": int(
                dict(schedule_row.get("generation_config", GENERATION_CONFIG))[
                    "completion_budget_tokens"
                ]
            ),
            "finish_reason": None,
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
    """Load one model once and durably acquire all missing rows in one shard."""

    if llama_factory is None:  # pragma: no cover - exercised by the live child.
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
        logits_all=True,
        seed=RANDOM_SEED,
        verbose=True,
    )
    raw_path = Path(str(payload["raw_path"]))
    checkpoint_path = Path(str(payload["checkpoint_path"]))
    manifest_hash = str(payload["manifest_hash"])
    existing = load_checkpoint(checkpoint_path, manifest_hash) if checkpoint_path.is_file() else []
    completed = {str(row["raw_key"]) for row in existing}
    receipts: list[JsonDict] = []
    try:
        metadata = deepcopy(dict(getattr(llm, "metadata", {}) or {}))
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
    return {
        "model_id": payload["model_id"],
        "phase": payload["phase"],
        "metadata": metadata,
        "metadata_hash": sha256_text(canonical_json(metadata)),
        "row_count": len(rows),
        "checkpoint_receipts": receipts,
        "model_close_called": True,
        "terminal_state": "complete",
    }


def parse_entrance(raw_text: str) -> JsonDict | None:
    """Extract the first exact JSON object and normalize its branch key."""

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text):
        try:
            value, _end = decoder.raw_decode(raw_text[match.start() :])
        except json.JSONDecodeError:
            continue
        pair = value.get("operand_pair")
        operator = value.get("operator")
        if (
            isinstance(pair, list)
            and len(pair) == 2
            and all(type(item) is int for item in pair)
            and operator in {"+", "-", "*", "/"}
        ):
            return {"operand_pair": sorted(pair), "operator": operator}
    return None


def _entrance_key(row: Mapping[str, Any]) -> tuple[str, tuple[int, int], str]:
    return (
        str(row.get("unit_id")),
        tuple(int(value) for value in row.get("operand_pair", [])),
        str(row.get("operator")),
    )


def label_proposal_rows(
    raw_rows: Sequence[Mapping[str, Any]], entrance_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Join raw proposals to the Exp7064 enumerator without changing raw rows."""

    exact = {_entrance_key(row): row for row in entrance_rows}
    seen: Counter[tuple[str, str, tuple[int, int], str]] = Counter()
    labels: list[JsonDict] = []
    for raw in raw_rows:
        parsed = parse_entrance(str(raw.get("raw_text", "")))
        exact_row = None
        duplicate = False
        if parsed is not None:
            key = (str(raw.get("unit_id")), tuple(parsed["operand_pair"]), parsed["operator"])
            exact_row = exact.get(key)
            duplicate_key = (
                str(raw.get("model_id")),
                str(raw.get("unit_id")),
                tuple(parsed["operand_pair"]),
                str(parsed["operator"]),
            )
            duplicate = seen[duplicate_key] > 0
            seen[duplicate_key] += 1
        labels.append(
            {
                **deepcopy(dict(raw)),
                "parsed_entrance": parsed,
                "entrance_id": exact_row.get("entrance_id") if exact_row else None,
                "parse_failure": parsed is None,
                "legal": exact_row is not None,
                "reachable": exact_row.get("reachable") is True if exact_row else False,
                "duplicate": duplicate,
                "label_source": "experiment_7064_exhaustive_enumerator",
            }
        )
    return labels


def _prefix_payload(entrance: Mapping[str, Any]) -> JsonDict:
    return {
        "left": int(entrance["left"]),
        "right": int(entrance["right"]),
        "operator": str(entrance["operator"]),
        "result": int(entrance["result"]),
    }


def select_forced_prefixes(
    model_ids: Sequence[str],
    unit_rows: Sequence[Mapping[str, Any]],
    entrance_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    diversity_unit_ids: Sequence[str],
    *,
    count_per_model: int = FORCED_UNIT_COUNT,
) -> list[JsonDict]:
    """Choose one reachable proposal-absent entrance per frozen diversity unit."""

    units = {str(row["unit_id"]): row for row in unit_rows}
    entrances: dict[str, list[Mapping[str, Any]]] = {}
    for row in entrance_rows:
        if row.get("reachable") is True:
            entrances.setdefault(str(row["unit_id"]), []).append(row)
    selected: list[JsonDict] = []
    for model_id in model_ids:
        proposed = {
            str(row.get("entrance_id"))
            for row in proposal_rows
            if row.get("model_id") == model_id and row.get("entrance_id")
        }
        added = 0
        for unit_id in diversity_unit_ids:
            if added >= count_per_model:
                break
            choices = sorted(entrances.get(unit_id, []), key=lambda row: str(row["entrance_id"]))
            entrance = next(
                (row for row in choices if str(row["entrance_id"]) not in proposed), None
            )
            if entrance is None or unit_id not in units:
                continue
            proposal_prompt = build_proposal_prompt(units[unit_id])
            prefix = _prefix_payload(entrance)
            prefix_json = canonical_json(prefix)
            forced_prompt = FORCED_TEMPLATE.format(
                proposal_prompt=proposal_prompt, prefix_json=prefix_json
            )
            selected.append(
                {
                    "model_id": model_id,
                    "unit_id": unit_id,
                    "entrance_id": entrance["entrance_id"],
                    "operand_pair": deepcopy(entrance["operand_pair"]),
                    "operator": entrance["operator"],
                    "left": entrance["left"],
                    "right": entrance["right"],
                    "result": entrance["result"],
                    "reachable": True,
                    "initially_unselected": True,
                    "prefix_json": prefix_json,
                    "prefix_hash": sha256_text(prefix_json),
                    "original_proposal_prompt": proposal_prompt,
                    "original_proposal_prompt_hash": sha256_text(proposal_prompt),
                    "prompt": forced_prompt,
                    "prompt_hash": sha256_text(forced_prompt),
                }
            )
            added += 1
    return selected


def build_forced_schedule(
    prefixes: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Add model paths and the matched remaining-budget contract."""

    paths = {str(row["hf_id"]): str(row["model_path"]) for row in specs}
    rows = []
    for prefix in prefixes:
        model_id = str(prefix["model_id"])
        rows.append(
            {
                **deepcopy(dict(prefix)),
                "raw_key": f"{model_id}|forced_prefix|{prefix['unit_id']}|{FORCED_PREFIX_SEED}",
                "model_path": paths[model_id],
                "seed": FORCED_PREFIX_SEED,
                "arm": "forced_prefix",
                "generation_config": deepcopy(GENERATION_CONFIG),
            }
        )
    random.Random(RANDOM_SEED + 1).shuffle(rows)
    for index, row in enumerate(rows):
        row["execution_index"] = index
    return rows


def _parse_step_array(raw_text: str) -> list[JsonDict] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", raw_text):
        try:
            value, _end = decoder.raw_decode(raw_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
            return [dict(row) for row in value]
    return None


def continuation_succeeds(
    unit: Mapping[str, Any], prefix: Mapping[str, Any], raw_text: str
) -> bool:
    """Replay the forced entrance and generated continuation through Exp7064."""

    steps = _parse_step_array(raw_text)
    if steps is None:
        return False
    normalized_steps = []
    for step in steps:
        normalized = deepcopy(step)
        if "operand_pair" not in normalized:
            try:
                normalized["operand_pair"] = sorted(
                    [int(normalized["left"]), int(normalized["right"])]
                )
            except (KeyError, TypeError, ValueError):
                return False
        normalized_steps.append(normalized)
    entrance = {
        "operand_pair": deepcopy(prefix["operand_pair"]),
        "operator": prefix["operator"],
        "reachable": True,
        "continuation_witness": normalized_steps,
    }
    return fixture_api.replay_entrance_witness(unit["numbers"], int(unit["target"]), entrance)


def label_forced_rows(
    raw_rows: Sequence[Mapping[str, Any]], unit_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Add exact continuation outcomes while retaining every raw field."""

    units = {str(row["unit_id"]): row for row in unit_rows}
    return [
        {
            **deepcopy(dict(row)),
            "parse_failure": _parse_step_array(str(row.get("raw_text", ""))) is None,
            "continuation_success": continuation_succeeds(
                units[str(row["unit_id"])], row, str(row.get("raw_text", ""))
            ),
            "label_source": "experiment_7064_exact_replay",
        }
        for row in raw_rows
    ]


def cleanup_passes(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Credit cleanup only when every signal, if any, targeted an owned PID."""

    return bool(rows) and all(
        row.get("process_owned") is True
        and row.get("owned_process_absent") is True
        and row.get("port_release_confirmed") is True
        and row.get("gpu_leases_released") is True
        and row.get("vram_release_passed") is True
        and (not row.get("signals_sent") or row.get("signaled_pid_owned") is True)
        for row in rows
    )


def unattributed_resource_gate(kind: str, observed: Any) -> JsonDict:
    """Describe a foreign resource without granting signal authority."""

    return gate_row(
        f"unattributed_{kind}_count",
        0,
        {"resources": deepcopy(observed), "signals_sent": []},
        False,
    )


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one expected-observed shape for every gate."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def _writable(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7065-write-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
        return True
    except OSError:
        return False


def _lease_probe(  # pragma: no cover - requires live kernel locks and process identity.
    devices: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Read journals and fail closed on unreadable or live foreign owners."""

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
        except Exception as exc:  # noqa: BLE001 - unreadable ownership must block.
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


def _stop_authority_probe() -> JsonDict:  # pragma: no cover - invokes the live host authority.
    """Run the shipped authority in dry-run mode so it cannot signal or write."""

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


def _identity_probe(  # pragma: no cover - reads a multi-gigabyte GGUF metadata header.
    model: Mapping[str, Any],
) -> JsonDict:
    """Load GGUF metadata only and bind its name to the declared family."""

    row = embedded_tokenizer_probe(model)
    metadata_text = canonical_json(row.get("metadata", {})).lower()
    expected = str(model["hf_id"]).rsplit("/", 1)[-1].removesuffix("-GGUF").lower()
    normalized = re.sub(r"[^a-z0-9]", "", expected)
    observed = re.sub(r"[^a-z0-9]", "", metadata_text)
    path_observed = re.sub(r"[^a-z0-9]", "", Path(str(model["model_path"])).name.lower())
    return {
        **row,
        "identity_matches": bool(
            row.get("passed") and normalized in observed and normalized in path_observed
        ),
        "expected_identity": expected,
        "tokenizer_source": "embedded_gguf",
    }


def collect_preconditions(
    *,
    fixture_path: Path,
    expected_fixture_hash: str,
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
    llama_probe: Callable[[], JsonDict] = llama_cpp_probe,
    lease_probe: Callable[[Sequence[Mapping[str, Any]]], list[JsonDict]] = _lease_probe,
    stop_authority_probe: Callable[[], JsonDict] = _stop_authority_probe,
    identity_probe: Callable[[Mapping[str, Any]], JsonDict] = _identity_probe,
) -> JsonDict:
    """Measure every upstream, model, CUDA, ownership, and path precondition."""

    fixture: JsonDict = {}
    fixture_hash = None
    fixture_valid = False
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        fixture_hash = sha256_file(fixture_path)
        fixture_valid = fixture_api.validate_artifact(fixture, check_files=True)
    except Exception as exc:  # noqa: BLE001 - exact failure text belongs in the gate.
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
            "entrance_fixture_ready_score",
            1,
            fixture.get("entrance_fixture_ready_score"),
            fixture_valid is True and fixture.get("entrance_fixture_ready_score") == 1,
        ),
        gate_row(
            "entrance_fixture_source_hash",
            expected_fixture_hash,
            fixture_hash,
            fixture_hash == expected_fixture_hash,
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
        "fixture": fixture,
        "fixture_hash": fixture_hash,
        "gpu_topology": gpu,
        "gpu_lease_preflight_rows": leases,
        "model_identity_rows": identities,
        "runner_build_rows": [llama],
        "stop_authority": stop,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Preserve every check and expose the first exact failure at top level."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def completion_errors(
    *,
    proposal_rows: Sequence[Mapping[str, Any]],
    forced_prefix_rows: Sequence[Mapping[str, Any]],
    held_unit_ids: Sequence[str],
    forced_unit_ids: Sequence[str],
    evidence: Mapping[str, Any],
) -> list[str]:
    """Recompute data completeness without reading proposal-quality labels."""

    errors: list[str] = []
    expected_proposals = {
        (model, unit, seed)
        for model in REQUIRED_MODEL_IDS
        for unit in held_unit_ids
        for seed in PROPOSAL_SEEDS
    }
    observed_proposals = {
        (str(row.get("model_id")), str(row.get("unit_id")), int(row.get("seed", -1)))
        for row in proposal_rows
    }
    if observed_proposals != expected_proposals or len(proposal_rows) != len(expected_proposals):
        errors.append("proposal_key_set_mismatch")
    expected_forced = {(model, unit) for model in REQUIRED_MODEL_IDS for unit in forced_unit_ids}
    observed_forced = {
        (str(row.get("model_id")), str(row.get("unit_id"))) for row in forced_prefix_rows
    }
    if observed_forced != expected_forced or len(forced_prefix_rows) != len(expected_forced):
        errors.append("forced_prefix_key_set_mismatch")
    all_rows = [*proposal_rows, *forced_prefix_rows]
    if any(
        row.get("terminal_state") != "complete"
        or row.get("raw_persisted_before_parse") is not True
        or row.get("raw_output_hash") != sha256_text(str(row.get("raw_text", "")))
        for row in all_rows
    ):
        errors.append("raw_terminal_or_hash_mismatch")
    hashes = list(evidence.get("raw_output_hashes", []))
    if hashes != [row.get("raw_output_hash") for row in all_rows]:
        errors.append("raw_output_hash_projection_mismatch")
    identity_ids = {
        str(row.get("model_id"))
        for row in evidence.get("model_identity_rows", [])
        if row.get("identity_matches") is True
    }
    if identity_ids != set(REQUIRED_MODEL_IDS):
        errors.append("model_identity_incomplete")
    file_hash_ids = {
        str(row.get("model_id"))
        for row in evidence.get("model_file_hash_rows", [])
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(row.get("sha256", "")))
    }
    if file_hash_ids != set(REQUIRED_MODEL_IDS):
        errors.append("model_file_hash_incomplete")
    cleanup_ids = {
        str(row.get("model_id"))
        for row in evidence.get("cleanup_rows", [])
        if row.get("passed") is True
    }
    if cleanup_ids != set(REQUIRED_MODEL_IDS):
        errors.append("cleanup_incomplete")
    if evidence.get("cuda_layer_offload_confirmed") is not True:
        errors.append("cuda_offload_unconfirmed")
    return errors


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the full terminal artifact except its self-referential digest."""

    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(stable))


def _per_game_results(
    proposal_rows: Sequence[Mapping[str, Any]], forced_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    unit_ids = sorted({str(row["unit_id"]) for row in proposal_rows})
    return [
        {
            "unit_id": unit_id,
            "proposal_count": sum(row.get("unit_id") == unit_id for row in proposal_rows),
            "legal_count": sum(
                row.get("unit_id") == unit_id and row.get("legal") is True for row in proposal_rows
            ),
            "reachable_count": sum(
                row.get("unit_id") == unit_id and row.get("reachable") is True
                for row in proposal_rows
            ),
            "duplicate_count": sum(
                row.get("unit_id") == unit_id and row.get("duplicate") is True
                for row in proposal_rows
            ),
            "parse_failure_count": sum(
                row.get("unit_id") == unit_id and row.get("parse_failure") is True
                for row in proposal_rows
            ),
            "forced_continuation_count": sum(row.get("unit_id") == unit_id for row in forced_rows),
            "forced_continuation_success_count": sum(
                row.get("unit_id") == unit_id and row.get("continuation_success") is True
                for row in forced_rows
            ),
        }
        for unit_id in unit_ids
    ]


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    proposal_rows: Sequence[Mapping[str, Any]] = (),
    forced_prefix_rows: Sequence[Mapping[str, Any]] = (),
    held_unit_ids: Sequence[str] = (),
    forced_unit_ids: Sequence[str] = (),
    selected_model_specs: Sequence[Mapping[str, Any]] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
    model_file_hash_rows: Sequence[Mapping[str, Any]] = (),
    raw_output_manifest: Sequence[Mapping[str, Any]] = (),
    checkpoint_rows: Sequence[Mapping[str, Any]] = (),
    phase_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one schema-complete blocked, partial, or positive artifact."""

    proposals = [deepcopy(dict(row)) for row in proposal_rows]
    forced = [deepcopy(dict(row)) for row in forced_prefix_rows]
    phases = [deepcopy(dict(row)) for row in phase_rows]
    identity_rows = deepcopy(list(preconditions.get("model_identity_rows", [])))
    cleanup_rows = [deepcopy(dict(row.get("cleanup", {}))) for row in phases]
    gpu_lease_rows = [
        deepcopy(dict(lease)) for row in phases for lease in row.get("gpu_lease_rows", [])
    ]
    port_lease_rows = [deepcopy(dict(row.get("port_lease", {}))) for row in phases]
    gpu_sample_rows = [
        deepcopy(dict(sample)) for row in phases for sample in row.get("gpu_sample_rows", [])
    ]
    cuda_confirmed = bool(
        phases
        and all(
            row.get("offloaded_layers", 0) > 0 and row.get("used_both_gpus") is True
            for row in phases
        )
    )
    raw_hashes = [row.get("raw_output_hash") for row in [*proposals, *forced]]
    evidence = {
        "model_identity_rows": identity_rows,
        "model_file_hash_rows": list(model_file_hash_rows),
        "cleanup_rows": cleanup_rows,
        "cuda_layer_offload_confirmed": cuda_confirmed,
        "raw_output_hashes": raw_hashes,
    }
    errors = (
        completion_errors(
            proposal_rows=proposals,
            forced_prefix_rows=forced,
            held_unit_ids=held_unit_ids,
            forced_unit_ids=forced_unit_ids,
            evidence=evidence,
        )
        if preconditions.get("all_passed") is True
        else ["preconditions_failed"]
    )
    complete = preconditions.get("all_passed") is True and not errors
    if preconditions.get("all_passed") is not True:
        verdict_class = "blocked"
        verdict = "blocked_v619_three_family_entrance_bank_precondition_failed"
    elif complete:
        verdict_class = "positive"
        verdict = "complete_positive_three_family_entrance_proposal_bank_acquired"
    else:
        verdict_class = "partial"
        verdict = "partial_three_family_entrance_proposal_bank_incomplete"
    gate_rows = [
        gate_row("model_family_count", 3, len(model_specs), len(model_specs) == 3),
        gate_row(
            "proposal_row_completeness",
            [],
            [e for e in errors if "proposal" in e],
            not any("proposal" in e for e in errors),
        ),
        gate_row(
            "forced_prefix_row_completeness",
            [],
            [e for e in errors if "forced" in e],
            not any("forced" in e for e in errors),
        ),
        gate_row(
            "raw_identity_cuda_cleanup",
            [],
            [e for e in errors if "proposal" not in e and "forced" not in e],
            not any("proposal" not in e and "forced" not in e for e in errors),
        ),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes or {})),
        "cited_upstream_artifacts": [
            {
                "path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
                "sha256": preconditions.get("fixture_hash"),
            }
        ],
        "upstream_gate_rows": [
            deepcopy(dict(row))
            for row in preconditions.get("checks", [])
            if row.get("check") in {"entrance_fixture_ready_score", "entrance_fixture_source_hash"}
        ],
        "rows": gate_rows,
        "per_game_results": _per_game_results(proposals, forced),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "models_used": list(REQUIRED_MODEL_IDS) if phases else [],
        "selected_model_specs": [deepcopy(dict(row)) for row in selected_model_specs],
        "model_identity_rows": identity_rows,
        "model_file_hash_rows": [deepcopy(dict(row)) for row in model_file_hash_rows],
        "runner_build_rows": deepcopy(list(preconditions.get("runner_build_rows", []))),
        "gpu_lease_rows": gpu_lease_rows,
        "port_lease_rows": port_lease_rows,
        "gpu_sample_rows": gpu_sample_rows,
        "cuda_layer_offload_confirmed": cuda_confirmed,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_hash": sha256_text(PROMPT_TEMPLATE),
        "matched_budget_contract": deepcopy(GENERATION_CONFIG),
        "proposal_rows": proposals,
        "forced_prefix_rows": forced,
        "raw_output_manifest": [deepcopy(dict(row)) for row in raw_output_manifest],
        "raw_output_hashes": raw_hashes,
        "checkpoint_rows": [deepcopy(dict(row)) for row in checkpoint_rows],
        "cleanup_rows": cleanup_rows,
        "model_family_count": len({row.get("hf_id") for row in model_specs}),
        "entrance_proposal_bank_complete_score": int(complete),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(preconditions.get("checks", []))
        if preconditions.get("all_passed") is not True
        else gate_summary(gate_rows),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, projections, terminal class, and content hash."""

    errors: list[str] = []
    errors.extend(
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    )
    if errors:
        return errors
    missing_principles = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {}))
    if missing_principles:
        errors.append("field_principles_mismatch")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    score = artifact.get("entrance_proposal_bank_complete_score")
    if type(score) is not int or score not in (0, 1):
        errors.append("completion_score_not_bare_int")
    if artifact.get("MODEL_SPECS") != artifact.get("model_specs"):
        errors.append("model_specs_projection_mismatch")
    if artifact.get("prompt_hash") != sha256_text(str(artifact.get("prompt_template", ""))):
        errors.append("prompt_hash_mismatch")
    proposals = list(artifact.get("proposal_rows", []))
    forced = list(artifact.get("forced_prefix_rows", []))
    if artifact.get("raw_output_hashes") != [
        row.get("raw_output_hash") for row in [*proposals, *forced]
    ]:
        errors.append("raw_output_hash_projection_mismatch")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    preflight = artifact.get("preconditions_checked", {}).get("all_passed") is True
    if not preflight:
        summary = artifact.get("gate_check_summary", {})
        if verdict_class != "blocked" or not verdict.startswith("blocked_") or score != 0:
            errors.append("blocked_verdict_mismatch")
        if not all(key in summary for key in ("failed_check", "expected_value", "observed_value")):
            errors.append("blocked_gate_summary_incomplete")
    elif score == 1:
        if verdict_class != "positive" or not verdict.startswith("complete_positive_"):
            errors.append("positive_verdict_mismatch")
    elif verdict_class != "partial" or not verdict.startswith("partial_"):
        errors.append("partial_verdict_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _source_hashes() -> JsonDict:  # pragma: no cover - hashes live repository files.
    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-references.md"),
        Path("openspec/capabilities/llm-ebm-inference/spec.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("python/carnot/experiment_7065_v619_three_family_entrance_bank.py"),
        Path("tests/python/test_experiment_7065_v619_three_family_entrance_bank.py"),
        Path("scripts/experiments/experiment_7065_v619_three_family_entrance_bank.py"),
        Path("scripts/run_stop_authority.py"),
    )
    rows = [
        {
            "path": str(path),
            "sha256": sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None,
        }
        for path in paths
    ]
    return {"files": rows, "manifest_hash": sha256_text(canonical_json(rows))}


def _choose_free_port() -> int:  # pragma: no cover - live kernel socket boundary.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _port_owner_pids(port: int) -> list[int]:  # pragma: no cover - live kernel socket boundary.
    result = subprocess.run(
        ["ss", "-Hlnpt", f"sport = :{int(port)}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return sorted({int(value) for value in re.findall(r"pid=(\d+)", result.stdout)})


def _port_is_free(port: int) -> bool:  # pragma: no cover - live kernel socket boundary.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False


def _memory_rows(  # pragma: no cover - projection of live nvidia-smi evidence.
    gpu: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        {
            "index": row.get("index"),
            "uuid": row.get("uuid"),
            "memory_used_mb": row.get("memory_used_mb"),
        }
        for row in gpu.get("devices", [])
    ]


def _wait_vram_release(  # pragma: no cover - polls live devices after model unload.
    baseline_rows: Sequence[Mapping[str, Any]], model_id: str
) -> JsonDict:
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


def _acquire_leases(  # pragma: no cover - acquires live kernel file locks.
    model: Mapping[str, Any], devices: Sequence[Mapping[str, Any]]
) -> tuple[list[Any], list[JsonDict]]:
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
            except Exception:  # noqa: BLE001 - best effort only for leases already owned.
                lease.close()
        raise
    return leases, rows


def _release_leases(  # pragma: no cover - terminalizes live lease journals.
    leases: Sequence[Any],
    rows: list[JsonDict],
    *,
    resident: bool,
    complete: bool,
    release: Mapping[str, Any],
    exit_code: int,
) -> None:
    after = {str(row["uuid"]): int(row["memory_used_mb"]) for row in release.get("after_rows", [])}
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
        except Exception as exc:  # noqa: BLE001 - release failure must remain visible.
            lease.close()
            row["release_error"] = f"{type(exc).__name__}: {exc}"


def run_model_phase(  # pragma: no cover - required live CUDA end-to-end boundary.
    *,
    model: Mapping[str, Any],
    phase: str,
    schedule_rows: Sequence[Mapping[str, Any]],
    devices: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    checkpoint_dir: Path,
) -> JsonDict:
    """Run one fresh owned worker and prove CUDA, port, lease, and VRAM release."""

    baseline = gpu_inventory()
    if baseline.get("processes"):
        return {
            "model_id": model["hf_id"],
            "phase": phase,
            "terminal_state": "blocked",
            "error": "unattributed_gpu_process",
            "cleanup": {"model_id": model["hf_id"], "passed": False},
        }
    baseline_rows = _memory_rows(baseline)
    leases, lease_rows = _acquire_leases(model, devices)
    slug = re.sub(r"[^a-z0-9]+", "-", str(model["hf_id"]).lower()).strip("-")
    phase_dir = raw_dir / f"{phase}-{slug}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    raw_path = phase_dir / "raw.jsonl"
    checkpoint_path = checkpoint_dir / f"{phase}-{slug}.json"
    payload_path = phase_dir / "payload.json"
    output_path = phase_dir / "worker-output.json"
    ready_path = phase_dir / "ready.json"
    stdout_path = phase_dir / "stdout.log"
    stderr_path = phase_dir / "stderr.log"
    manifest_hash = sha256_text(
        canonical_json(
            {
                "model": model,
                "phase": phase,
                "rows": list(schedule_rows),
                "config": GENERATION_CONFIG,
            }
        )
    )
    write_json_atomic(
        payload_path,
        {
            "model_id": model["hf_id"],
            "model_path": model["model_path"],
            "phase": phase,
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
    samples: list[JsonDict] = []
    ready_owned = False
    resident = False
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
                ready_owned = bool(
                    ready.get("pid") == process.pid
                    and ready.get("pid_start_ticks") == start_ticks
                    and ready.get("port") == port
                    and _port_owner_pids(port) == [process.pid]
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
        signals: list[str] = []
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
        "phase": phase,
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
        "phase": phase,
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
        "gpu_lease_rows": lease_rows,
        "port_lease": {
            "model_id": model["hf_id"],
            "phase": phase,
            "port": port,
            "owned_by_worker": ready_owned,
            "released": port_free,
        },
        "cleanup": cleanup,
        "vram_release": release,
        "backend_stderr_hash": sha256_text(stderr_text),
    }


def _worker_main(  # pragma: no cover - private subprocess and readiness-socket boundary.
    payload_path: Path, output_path: Path, ready_path: Path, port: int
) -> int:
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
    except Exception as exc:  # noqa: BLE001 - parent must retain exact worker failure.
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


def run(  # pragma: no cover - required live six-process experiment controller.
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    fixture_path: Path = FIXTURE_PATH,
    raw_dir: Path = RAW_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Preflight, acquire proposal shards, label, acquire forced shards, and validate."""

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
    if preconditions.get("all_passed") is not True:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            model_specs=specs,
            preconditions=preconditions,
            source_artifact_hashes=source_hashes,
        )
        write_json_atomic(result_path, artifact)
        return artifact

    fixture = preconditions["fixture"]
    held_groups = set(fixture["split_manifest"]["held_source_group_ids"])
    held_unit_ids = [
        str(row["unit_id"]) for row in fixture["unit_rows"] if row["source_group_id"] in held_groups
    ]
    visible = fixture["model_visible_rows"]
    proposal_schedule = build_proposal_schedule(specs, visible, held_unit_ids)
    if matched_schedule_errors(proposal_schedule):
        raise RuntimeError(
            f"proposal_schedule_invalid:{matched_schedule_errors(proposal_schedule)}"
        )
    rng = random.Random(RANDOM_SEED)
    model_order = list(REQUIRED_MODEL_IDS)
    rng.shuffle(model_order)
    selected_specs = [
        next(row for row in specs if row["hf_id"] == model_id) for model_id in model_order
    ]
    devices = preconditions["gpu_topology"]["devices"]
    phase_rows: list[JsonDict] = []
    for model in selected_specs:
        phase_rows.append(
            run_model_phase(
                model=model,
                phase="proposal",
                schedule_rows=[
                    row for row in proposal_schedule if row["model_id"] == model["hf_id"]
                ],
                devices=devices,
                raw_dir=raw_dir,
                checkpoint_dir=checkpoint_dir,
            )
        )
        if phase_rows[-1]["terminal_state"] != "complete":
            break
    proposal_raw = [row for phase in phase_rows for row in phase.get("raw_rows", [])]
    proposal_raw.sort(key=lambda row: str(row["raw_key"]))
    proposal_rows = label_proposal_rows(proposal_raw, fixture["entrance_rows"])
    diversity_held = [unit for unit in fixture["diversity_subset_ids"] if unit in held_unit_ids]
    prefixes = select_forced_prefixes(
        REQUIRED_MODEL_IDS,
        fixture["unit_rows"],
        fixture["entrance_rows"],
        proposal_rows,
        diversity_held,
    )
    forced_schedule = build_forced_schedule(prefixes, specs)
    if len(prefixes) == len(REQUIRED_MODEL_IDS) * FORCED_UNIT_COUNT and len(phase_rows) == 3:
        forced_order = list(selected_specs)
        random.Random(RANDOM_SEED + 1).shuffle(forced_order)
        for model in forced_order:
            phase_rows.append(
                run_model_phase(
                    model=model,
                    phase="forced_prefix",
                    schedule_rows=[
                        row for row in forced_schedule if row["model_id"] == model["hf_id"]
                    ],
                    devices=devices,
                    raw_dir=raw_dir,
                    checkpoint_dir=checkpoint_dir,
                )
            )
            if phase_rows[-1]["terminal_state"] != "complete":
                break
    forced_raw = [
        row
        for phase in phase_rows
        if phase.get("phase") == "forced_prefix"
        for row in phase.get("raw_rows", [])
    ]
    forced_raw.sort(key=lambda row: str(row["raw_key"]))
    forced_rows = label_forced_rows(forced_raw, fixture["unit_rows"])
    forced_unit_ids = diversity_held[:FORCED_UNIT_COUNT]
    model_hash_rows = [
        {
            "model_id": row["hf_id"],
            "path": row["model_path"],
            "sha256": sha256_file(row["model_path"]),
        }
        for row in specs
    ]
    raw_manifest = [
        {
            "model_id": row["model_id"],
            "phase": row["phase"],
            "path": row.get("raw_path"),
            "sha256": row.get("raw_sha256"),
            "row_count": len(row.get("raw_rows", [])),
        }
        for row in phase_rows
    ]
    checkpoint_rows = [
        {
            "model_id": row["model_id"],
            "phase": row["phase"],
            "path": row.get("checkpoint_path"),
            "sha256": row.get("checkpoint_sha256"),
            "manifest_hash": row.get("manifest_hash"),
            "row_count": len(row.get("raw_rows", [])),
        }
        for row in phase_rows
    ]
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions=preconditions,
        proposal_rows=proposal_rows,
        forced_prefix_rows=forced_rows,
        held_unit_ids=held_unit_ids,
        forced_unit_ids=forced_unit_ids,
        selected_model_specs=selected_specs,
        source_artifact_hashes=source_hashes,
        model_file_hash_rows=model_hash_rows,
        raw_output_manifest=raw_manifest,
        checkpoint_rows=checkpoint_rows,
        phase_rows=phase_rows,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(  # pragma: no cover - exercised by the required live command.
    argv: Sequence[str] | None = None,
) -> int:
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
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
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
                "entrance_proposal_bank_complete_score": artifact[
                    "entrance_proposal_bank_complete_score"
                ],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - module command surface.
    raise SystemExit(main())
