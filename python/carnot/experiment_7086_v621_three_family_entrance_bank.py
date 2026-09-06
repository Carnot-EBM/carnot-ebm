"""Acquire a complete three-family entrance bank through GGUF chat templates.

Spec refs: REQ-INFRA-7086, REQ-VERIFY-7086, and their scenarios.

The worker makes the chat call that Exp7085 proved. It checkpoints raw bytes
before the controller opens the Exp7064 exact labels. The exact enumerator
measures arithmetic outcomes but does not control generation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback
from typing import Any

from carnot import experiment_7080_v620_three_family_entrance_bank as prior_bank
from carnot import experiment_7085_v621_chat_transport_canary as chat_canary
from carnot import gpu_lease_phase_journal as lease_api
from carnot.task_runtime_receipts import sha256_file, write_json_atomic


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "carnot.experiment_7086_v621_three_family_entrance_bank"
EXPERIMENT_ID = "experiment_7086_v621_three_family_entrance_bank"
SCHEMA = "carnot.experiment_7086.v621_three_family_entrance_bank.v1"
RUN_DATE = "20260906"
RANDOM_SEED = 7_086_202_609_06
PROPOSAL_SEEDS = (7_086_001, 7_086_002, 7_086_003, 7_086_004)
FORCED_PREFIX_SEED = 7_086_101
FORCED_UNIT_COUNT = 24
INFERENCE_SUBSTRATE = "full live local SOTA GGUF chat generation"
RESULT_PATH = REPO_ROOT / "results/experiment_7086_v621_three_family_entrance_bank.json"
FIXTURE_PATH = REPO_ROOT / "results/experiment_7064_v619_exact_entrance_fixture.json"
LEASE_AUDIT_PATH = REPO_ROOT / "results/experiment_7079_v620_gpu_lease_audit.json"
PRIOR_BANK_PATH = REPO_ROOT / "results/experiment_7080_v620_three_family_entrance_bank.json"
CHAT_CANARY_PATH = REPO_ROOT / "results/experiment_7085_v621_chat_transport_canary.json"
RAW_DIR = REPO_ROOT / "results/raw/experiment_7086_v621_three_family_entrance_bank"
CHECKPOINT_DIR = REPO_ROOT / "results/checkpoints/experiment_7086_v621_three_family_entrance_bank"
PINNED_FIXTURE_SHA256 = "sha256:6b62768e3387d40eebf462c199aab6a440321aa4a1549ff7d54faaba312f2277"
PINNED_CHAT_CANARY_SHA256 = (
    "sha256:ae0fb899b82f5ec357de12dce12c07d8ef1d3bc651d2503388e43317efab0424"
)
PREFERRED_QUANT = "Q4_K_M"
MODEL_TIMEOUT_S = 7_200.0
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))

REQUIRED_MODEL_IDS = chat_canary.REQUIRED_MODEL_IDS
COMPLETE_PHASE_SEQUENCE = lease_api.COMPLETE_PHASE_SEQUENCE
GENERATION_CONFIG: JsonDict = deepcopy(chat_canary.GENERATION_CONFIG)
SYSTEM_MESSAGE = chat_canary.SYSTEM_MESSAGE
PROMPT_TEMPLATE = chat_canary.PROMPT_TEMPLATE
FORCED_TEMPLATE = (
    "{proposal_prompt}\nThe first branch is forced to {prefix_json}. Continue from that result "
    "under the same arithmetic rules. Return only one JSON array of the remaining operations. "
    "Each operation must contain left, right, operator, and result. Do not repeat the forced branch."
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
    "upstream_gate_rows",
    "rows",
    "per_game_results",
    "model_specs",
    "model_identity_rows",
    "model_execution_rows",
    "chat_template_rows",
    "rendered_prompt_hash_rows",
    "stop_config_rows",
    "runner_build_rows",
    "raw_proposal_rows",
    "token_score_rows",
    "token_count_rows",
    "finish_reason_rows",
    "parse_rows",
    "exact_label_rows",
    "causal_witness_rows",
    "guess_without_witness_rows",
    "forced_prefix_rows",
    "per_model_rows",
    "per_source_group_rows",
    "prompt_hash",
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
    "gpu_topology_rows",
    "gpu_lease_rows",
    "vram_release_rows",
    "signals_sent",
    "empty_output_rate_by_model",
    "zero_token_rate_by_model",
    "parseable_rate_by_model",
    "leaked_control_token_count_by_model",
    "length_limited_count_by_model",
    "all_models_real",
    "entrance_proposal_bank_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

_EXTRA_FIELD_PRINCIPLES = {
    "inference_substrate_class": "The class distinguishes a real model run from a preflight block.",
    "chat_template_rows": "Template receipts detect instruction-transport drift.",
    "rendered_prompt_hash_rows": "Rendered hashes detect backend prompt changes.",
    "stop_config_rows": "Stop receipts expose hidden termination differences.",
    "token_count_rows": "Token counts detect immediate end-of-sequence output.",
    "finish_reason_rows": "Finish reasons expose budget truncation.",
    "causal_witness_rows": "Exact witnesses distinguish reachable branches from unsupported guesses.",
    "guess_without_witness_rows": "Guess controls prevent answer-only success from receiving causal credit.",
    "runner_receipt": "The runner receipt binds output to llama.cpp chat generation.",
    "generation_invoked": "This flag separates attempted inference from a blocked preflight.",
    "total_model_count": "The distinct count detects a missing family.",
    "model_load_count_by_stage": "Load counts detect hidden reloads and missing shards.",
    "per_model_duration_s": "Family durations expose omitted or stalled execution.",
    "stage_gpu_telemetry_rows": "Stage samples connect each shard to GPU residence.",
    "task_gpu_telemetry_rows": "Task samples preserve the full resource boundary.",
    "peak_vram_by_device": "Peak memory provides independent model-residence evidence.",
    "empty_output_rate_by_model": "Family-local empty rates detect broken output transport.",
    "zero_token_rate_by_model": "Zero-token rates detect immediate termination.",
    "parseable_rate_by_model": "Parseability measures instruction transport without scoring arithmetic quality.",
    "leaked_control_token_count_by_model": "Control-token counts detect chat-boundary leakage.",
    "length_limited_count_by_model": "Length counts detect an insufficient output budget.",
}
FIELD_PRINCIPLES = {
    field: _EXTRA_FIELD_PRINCIPLES.get(
        field,
        chat_canary.FIELD_PRINCIPLES.get(
            field,
            prior_bank.FIELD_PRINCIPLES.get(
                field, f"Independent {field} evidence makes this result falsifiable."
            ),
        ),
    )
    for field in REQUIRED_ARTIFACT_FIELDS
}

canonical_json = chat_canary.canonical_json
sha256_text = chat_canary.sha256_text
gate_row = chat_canary.gate_row
gate_summary = chat_canary.gate_summary
resolve_model_specs = chat_canary.resolve_model_specs
model_spec_errors = chat_canary.model_spec_errors
model_identity_errors = chat_canary.model_identity_errors
build_proposal_prompt = chat_canary.build_proposal_prompt
build_role_messages = chat_canary.build_role_messages
load_checkpoint = chat_canary.load_checkpoint
checkpoint_raw_row = chat_canary.checkpoint_raw_row
_append_jsonl = chat_canary._append_jsonl
parse_entrance = prior_bank.parse_entrance
label_proposal_rows = prior_bank.label_proposal_rows
label_forced_rows = prior_bank.label_forced_rows
continuation_succeeds = prior_bank.continuation_succeeds
cleanup_passes = prior_bank.cleanup_passes
unattributed_resource_gate = prior_bank.unattributed_resource_gate


def build_proposal_schedule(
    specs: Sequence[Mapping[str, Any]],
    visible_units: Sequence[Mapping[str, Any]],
    ordered_unit_ids: Sequence[str],
    *,
    seed: int = RANDOM_SEED,
) -> list[JsonDict]:
    """Build every matched proposal cell, then randomize only execution order."""

    visible = {str(row["unit_id"]): row for row in visible_units}
    rows = []
    for unit_index, unit_id in enumerate(ordered_unit_ids):
        prompt = build_proposal_prompt(visible[unit_id])
        messages = build_role_messages(prompt)
        for proposal_seed in PROPOSAL_SEEDS:
            for model in specs:
                model_id = str(model["hf_id"])
                rows.append(
                    {
                        "raw_key": f"{model_id}|proposal|{unit_id}|{proposal_seed}",
                        "model_id": model_id,
                        "model_path": str(model["model_path"]),
                        "unit_id": unit_id,
                        "fixture_unit_index": unit_index,
                        "seed": proposal_seed,
                        "arm": "proposal",
                        "prompt": prompt,
                        "prompt_hash": sha256_text(prompt),
                        "role_messages": deepcopy(messages),
                        "role_messages_hash": sha256_text(canonical_json(messages)),
                        "generation_config": deepcopy(GENERATION_CONFIG),
                    }
                )
    random.Random(int(seed)).shuffle(rows)
    for execution_index, row in enumerate(rows):
        row["execution_index"] = execution_index
    return rows


def matched_schedule_errors(
    rows: Sequence[Mapping[str, Any]], ordered_unit_ids: Sequence[str]
) -> list[str]:
    """Reject missing cells or any cross-family prompt and chat-contract drift."""

    errors = []
    expected = {
        (model_id, unit_id, proposal_seed)
        for model_id in REQUIRED_MODEL_IDS
        for unit_id in ordered_unit_ids
        for proposal_seed in PROPOSAL_SEEDS
    }
    observed = {
        (str(row.get("model_id")), str(row.get("unit_id")), int(row.get("seed", -1)))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        errors.append("schedule_key_set_mismatch")
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        prompt = str(row.get("prompt", ""))
        messages = row.get("role_messages")
        if row.get("prompt_hash") != sha256_text(prompt):
            errors.append("prompt_hash_mismatch")
        if messages != build_role_messages(prompt):
            errors.append("role_message_mismatch")
        if row.get("role_messages_hash") != sha256_text(canonical_json(messages)):
            errors.append("role_message_hash_mismatch")
        if row.get("generation_config") != GENERATION_CONFIG:
            errors.append("generation_config_mismatch")
        grouped.setdefault((str(row.get("unit_id")), int(row.get("seed", -1))), []).append(row)
    for (unit_id, proposal_seed), group in grouped.items():
        if len({str(row.get("prompt")) for row in group}) != 1:
            errors.append(f"cross_model_prompt_mismatch:{unit_id}:{proposal_seed}")
        if len({canonical_json(row.get("generation_config")) for row in group}) != 1:
            errors.append(f"cross_model_budget_mismatch:{unit_id}:{proposal_seed}")
    return list(dict.fromkeys(errors))


def randomized_phase_order(seed: int = RANDOM_SEED) -> list[JsonDict]:
    """Randomize model order independently inside the two dependent arms."""

    rows = []
    for arm_index, phase in enumerate(("proposal", "forced_prefix")):
        model_ids = list(REQUIRED_MODEL_IDS)
        random.Random(int(seed) + arm_index).shuffle(model_ids)
        rows.extend(
            {
                "phase": phase,
                "phase_order_index": index,
                "model_id": model_id,
                "random_seed": int(seed) + arm_index,
            }
            for index, model_id in enumerate(model_ids)
        )
    return rows


def worker_generate_one(
    schedule_row: Mapping[str, Any],
    *,
    llama_factory: Callable[..., Any] | None = None,
    llama_instance: Any = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Use the approved chat call and account for a forced prefix budget."""

    owns_llama = llama_instance is None
    llm = llama_instance
    config = deepcopy(dict(schedule_row.get("generation_config") or GENERATION_CONFIG))
    if llm is None:  # pragma: no cover - the live shard loads once in worker_run_schedule.
        if llama_factory is None:
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
    prefix_count = 0
    adjusted = deepcopy(dict(schedule_row))
    if adjusted.get("arm") == "forced_prefix":
        prefix = str(adjusted.get("prefix_json", ""))
        prefix_count = len(llm.tokenize(prefix.encode("utf-8"), add_bos=False))
        config["completion_budget_tokens"] = max(
            1, int(GENERATION_CONFIG["completion_budget_tokens"]) - prefix_count
        )
    adjusted["generation_config"] = config
    try:
        row = chat_canary.worker_generate_one(
            adjusted,
            llama_instance=llm,
            clock=clock,
        )
    finally:
        if owns_llama:  # pragma: no cover - worker_run_schedule owns the live model.
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            llm = None
            gc.collect()
    row["generation_config"] = deepcopy(GENERATION_CONFIG)
    row["prefix_token_count"] = prefix_count
    row["requested_completion_budget_tokens"] = int(GENERATION_CONFIG["completion_budget_tokens"])
    row["effective_completion_budget_tokens"] = int(config["completion_budget_tokens"])
    row["token_scores"] = []
    row["token_scores_available"] = False
    row["model_close_called"] = bool(owns_llama)
    return row


def worker_run_schedule(
    payload: Mapping[str, Any], *, llama_factory: Callable[..., Any] | None = None
) -> JsonDict:
    """Load one GGUF once and checkpoint each missing raw unit before parsing."""

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
        seed=RANDOM_SEED,
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
    template = chat_canary._template_text(metadata)
    phase = str(payload.get("phase") or payload["schedule_rows"][0]["arm"])
    return {
        "model_id": payload["model_id"],
        "phase": phase,
        "metadata_hash": sha256_text(canonical_json(metadata)),
        "chat_template_present": bool(template),
        "chat_template_hash": sha256_text(template) if template else None,
        "row_count": len(rows),
        "checkpoint_receipts": receipts,
        "model_close_called": True,
        "terminal_state": "complete",
    }


def proposal_evidence_rows(
    proposal_rows: Sequence[Mapping[str, Any]],
    entrance_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Separate immutable raw rows from parse, exact, and causal projections."""

    del entrance_rows
    views = prior_bank.proposal_evidence_rows(proposal_rows)
    causal_rows = []
    guess_rows = []
    for label in views["exact_label_rows"]:
        causal = bool(label.get("reachable") is True and label.get("entrance_id"))
        identity = {
            "raw_key": label.get("raw_key"),
            "raw_output_hash": label.get("raw_output_hash"),
            "model_id": label.get("model_id"),
            "unit_id": label.get("unit_id"),
            "seed": label.get("seed"),
            "entrance_id": label.get("entrance_id"),
        }
        causal_rows.append(
            {
                **identity,
                "causal_witness": causal,
                "witness_source": "experiment_7064_exhaustive_enumerator" if causal else None,
            }
        )
        guess_rows.append(
            {**identity, "guess_without_witness": bool(label.get("legal") and not causal)}
        )
    return {
        **views,
        "causal_witness_rows": causal_rows,
        "guess_without_witness_rows": guess_rows,
    }


def select_diversity_unit_ids(
    unit_rows: Sequence[Mapping[str, Any]], *, count: int = FORCED_UNIT_COUNT
) -> list[str]:
    """Choose forced units round-robin so every source group remains visible."""

    groups: dict[str, list[str]] = {}
    for row in unit_rows:
        groups.setdefault(str(row["source_group_id"]), []).append(str(row["unit_id"]))
    selected = []
    depth = 0
    while len(selected) < count and any(depth < len(rows) for rows in groups.values()):
        for group in sorted(groups):
            if depth < len(groups[group]) and len(selected) < count:
                selected.append(groups[group][depth])
        depth += 1
    if len(selected) != count:
        raise ValueError(f"forced_diversity_unit_count:{len(selected)}")
    return selected


def select_forced_prefixes(
    model_ids: Sequence[str],
    unit_rows: Sequence[Mapping[str, Any]],
    entrance_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    diversity_unit_ids: Sequence[str],
    *,
    count_per_model: int = FORCED_UNIT_COUNT,
) -> list[JsonDict]:
    """Choose one reachable proposal-absent entrance for each forced cell."""

    units = {str(row["unit_id"]): row for row in unit_rows}
    entrances: dict[str, list[Mapping[str, Any]]] = {}
    for row in entrance_rows:
        if row.get("reachable") is True:
            entrances.setdefault(str(row["unit_id"]), []).append(row)
    selected = []
    for model_id in model_ids:
        proposed = {
            str(row.get("entrance_id"))
            for row in proposal_rows
            if row.get("model_id") == model_id and row.get("entrance_id")
        }
        for unit_id in list(diversity_unit_ids)[:count_per_model]:
            choices = sorted(entrances.get(unit_id, []), key=lambda row: str(row["entrance_id"]))
            entrance = next(
                (row for row in choices if str(row["entrance_id"]) not in proposed),
                None,
            )
            if entrance is None or unit_id not in units:
                continue
            unit = units[unit_id]
            proposal_prompt = build_proposal_prompt(unit)
            prefix = prior_bank._prefix_payload(entrance)
            prefix_json = canonical_json(prefix)
            prompt = FORCED_TEMPLATE.format(
                proposal_prompt=proposal_prompt,
                prefix_json=prefix_json,
            )
            messages = build_role_messages(prompt)
            selected.append(
                {
                    "model_id": model_id,
                    "unit_id": unit_id,
                    "source_group_id": str(unit["source_group_id"]),
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
                    "prompt": prompt,
                    "prompt_hash": sha256_text(prompt),
                    "role_messages": messages,
                    "role_messages_hash": sha256_text(canonical_json(messages)),
                }
            )
    return selected


def build_forced_schedule(
    prefixes: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    *,
    seed: int = RANDOM_SEED + 1,
) -> list[JsonDict]:
    """Attach local model paths and the unchanged Exp7085 chat contract."""

    paths = {str(row["hf_id"]): str(row["model_path"]) for row in specs}
    rows = []
    for prefix in prefixes:
        row = {
            **deepcopy(dict(prefix)),
            "raw_key": (
                f"{prefix['model_id']}|forced_prefix|{prefix['unit_id']}|{FORCED_PREFIX_SEED}"
            ),
            "model_path": paths[str(prefix["model_id"])],
            "seed": FORCED_PREFIX_SEED,
            "arm": "forced_prefix",
            "generation_config": deepcopy(GENERATION_CONFIG),
        }
        messages = build_role_messages(str(row["prompt"]))
        row["role_messages"] = messages
        row["role_messages_hash"] = sha256_text(canonical_json(messages))
        rows.append(row)
    random.Random(int(seed)).shuffle(rows)
    for execution_index, row in enumerate(rows):
        row["execution_index"] = execution_index
    return rows


def _row_metrics(
    proposal_rows: Sequence[Mapping[str, Any]], forced_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Compute transport rates independently for each model family."""

    all_rows = [*proposal_rows, *forced_rows]
    empty = {}
    zero = {}
    parseable = {}
    leaked = {}
    limited = {}
    per_model = []
    for model_id in REQUIRED_MODEL_IDS:
        rows = [row for row in all_rows if row.get("model_id") == model_id]
        count = len(rows)
        empty_count = sum(not str(row.get("raw_text", "")).strip() for row in rows)
        zero_count = sum(int(row.get("completion_tokens", 0) or 0) == 0 for row in rows)
        parse_count = sum(
            (
                prior_bank._parse_step_array(str(row.get("raw_text", ""))) is not None
                if row.get("arm") == "forced_prefix"
                else parse_entrance(str(row.get("raw_text", ""))) is not None
            )
            for row in rows
        )
        leak_count = sum(
            bool(chat_canary.CONTROL_TOKEN_RE.search(str(row.get("raw_text", "")))) for row in rows
        )
        length_count = sum(
            str(row.get("finish_reason", "")).lower() == "length"
            or int(row.get("completion_tokens", 0) or 0)
            > int(row.get("effective_completion_budget_tokens", 0) or 0)
            for row in rows
        )
        empty[model_id] = empty_count / count if count else 1.0
        zero[model_id] = zero_count / count if count else 1.0
        parseable[model_id] = parse_count / count if count else 0.0
        leaked[model_id] = leak_count
        limited[model_id] = length_count
        per_model.append(
            {
                "model_id": model_id,
                "generation_count": count,
                "empty_output_count": empty_count,
                "zero_token_count": zero_count,
                "parseable_count": parse_count,
                "leaked_control_token_count": leak_count,
                "length_limited_count": length_count,
            }
        )
    return {
        "empty_output_rate_by_model": empty,
        "zero_token_rate_by_model": zero,
        "parseable_rate_by_model": parseable,
        "leaked_control_token_count_by_model": leaked,
        "length_limited_count_by_model": limited,
        "per_model_transport_rows": per_model,
    }


def _expected_shards() -> set[tuple[str, str]]:
    return {
        (model_id, phase)
        for model_id in REQUIRED_MODEL_IDS
        for phase in ("proposal", "forced_prefix")
    }


def completion_errors(
    *,
    proposal_rows: Sequence[Mapping[str, Any]],
    forced_prefix_rows: Sequence[Mapping[str, Any]],
    ordered_unit_ids: Sequence[str],
    forced_unit_ids: Sequence[str],
    unit_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> list[str]:
    """Recompute complete acquisition and chat transport without quality labels."""

    errors = list(model_spec_errors(model_specs))
    errors.extend(model_identity_errors(model_specs, identity_rows))
    expected_proposals = {
        (model_id, unit_id, seed)
        for model_id in REQUIRED_MODEL_IDS
        for unit_id in ordered_unit_ids
        for seed in PROPOSAL_SEEDS
    }
    observed_proposals = {
        (str(row.get("model_id")), str(row.get("unit_id")), int(row.get("seed", -1)))
        for row in proposal_rows
    }
    if observed_proposals != expected_proposals or len(proposal_rows) != len(expected_proposals):
        errors.append("proposal_key_set_mismatch")
    expected_forced = {
        (model_id, unit_id) for model_id in REQUIRED_MODEL_IDS for unit_id in forced_unit_ids
    }
    observed_forced = {
        (str(row.get("model_id")), str(row.get("unit_id"))) for row in forced_prefix_rows
    }
    if observed_forced != expected_forced or len(forced_prefix_rows) != len(expected_forced):
        errors.append("forced_prefix_key_set_mismatch")
    groups = {str(row.get("source_group_id")) for row in unit_rows}
    forced_groups = {
        str(row.get("source_group_id"))
        for row in unit_rows
        if str(row.get("unit_id")) in set(forced_unit_ids)
    }
    if len(ordered_unit_ids) != 96 or len(groups) != 12:
        errors.append("fixture_panel_mismatch")
    if len(forced_unit_ids) < FORCED_UNIT_COUNT or forced_groups != groups:
        errors.append("forced_source_group_coverage_mismatch")
    all_rows = [*proposal_rows, *forced_prefix_rows]
    for row in all_rows:
        text = str(row.get("raw_text", ""))
        if (
            row.get("terminal_state") != "complete"
            or row.get("raw_persisted_before_parse") is not True
            or row.get("raw_output_hash") != sha256_text(text)
            or row.get("raw_bytes_hex") != text.encode("utf-8").hex()
        ):
            errors.append("raw_terminal_or_hash_mismatch")
        if row.get("transport_method") != "create_chat_completion":
            errors.append("chat_transport_mismatch")
        if (
            row.get("chat_template_present") is not True
            or not row.get("chat_template_hash")
            or not row.get("rendered_prompt_hash")
        ):
            errors.append("chat_template_receipt_missing")
        if row.get("role_messages") != build_role_messages(str(row.get("prompt", ""))):
            errors.append("role_message_mismatch")
        if row.get("prompt_hash") != sha256_text(str(row.get("prompt", ""))):
            errors.append("prompt_hash_mismatch")
        if row.get("stop_config") != GENERATION_CONFIG["stop"]:
            errors.append("stop_config_mismatch")
        if row.get("generation_config") != GENERATION_CONFIG:
            errors.append("sampling_config_mismatch")
        requested = int(row.get("requested_completion_budget_tokens", 0) or 0)
        effective = int(row.get("effective_completion_budget_tokens", 0) or 0)
        prefix_count = int(row.get("prefix_token_count", 0) or 0)
        if requested != 192 or effective + prefix_count != 192:
            errors.append("completion_budget_mismatch")
    if any(
        row.get("reachable") is not True or row.get("initially_unselected") is not True
        for row in forced_prefix_rows
    ):
        errors.append("forced_prefix_contract_mismatch")
    file_ids = {
        str(row.get("model_id"))
        for row in evidence.get("model_file_hash_rows", [])
        if str(row.get("sha256", "")).startswith("sha256:")
    }
    if file_ids != set(REQUIRED_MODEL_IDS):
        errors.append("model_file_hash_incomplete")
    shards = _expected_shards()
    execution_rows = list(evidence.get("model_execution_rows", []))
    if {
        (str(row.get("model_id")), str(row.get("phase"))) for row in execution_rows
    } != shards or any(
        row.get("terminal_state") != "complete"
        or int(row.get("raw_row_count", 0) or 0) <= 0
        or int(row.get("offloaded_layers", 0) or 0) <= 0
        or row.get("used_both_gpus") is not True
        or row.get("cleanup_passed") is not True
        or int(row.get("model_load_count", 0) or 0) != 1
        for row in execution_rows
    ):
        errors.append("model_execution_incomplete")
    cleanup_rows = list(evidence.get("cleanup_rows", []))
    if {
        (str(row.get("model_id")), str(row.get("phase"))) for row in cleanup_rows
    } != shards or not cleanup_passes(cleanup_rows):
        errors.append("cleanup_incomplete")
    leases = list(evidence.get("gpu_lease_rows", []))
    if (
        len(leases) != 12
        or {(str(row.get("model_id")), str(row.get("phase"))) for row in leases} != shards
        or any(
            not row.get("lease_id")
            or row.get("owner_preserved") is not True
            or list(row.get("phase_history", [])) != list(COMPLETE_PHASE_SEQUENCE)
            or row.get("released") is not True
            or row.get("lease_lost") is not False
            for row in leases
        )
    ):
        errors.append("lease_identity_or_release_incomplete")
    checkpoints = list(evidence.get("checkpoint_rows", []))
    if {(str(row.get("model_id")), str(row.get("phase"))) for row in checkpoints} != shards or any(
        not str(row.get("sha256", "")).startswith("sha256:")
        or not str(row.get("manifest_hash", "")).startswith("sha256:")
        or int(row.get("row_count", 0) or 0) <= 0
        for row in checkpoints
    ):
        errors.append("checkpoint_incomplete")
    releases = list(evidence.get("vram_release_rows", []))
    if {(str(row.get("model_id")), str(row.get("phase"))) for row in releases} != shards or any(
        row.get("passed") is not True for row in releases
    ):
        errors.append("vram_release_incomplete")
    for name in ("stage_gpu_telemetry_rows", "task_gpu_telemetry_rows"):
        rows = list(evidence.get(name, []))
        if {(str(row.get("model_id")), str(row.get("phase"))) for row in rows} != shards:
            errors.append(name.replace("_rows", "_incomplete"))
    peaks = dict(evidence.get("peak_vram_by_device", {}))
    if len(peaks) != 2 or any(int(value or 0) <= 0 for value in peaks.values()):
        errors.append("peak_vram_incomplete")
    runner = dict(evidence.get("runner_receipt") or {})
    if (
        runner.get("cuda_offload") is not True
        or runner.get("transport_method") != "create_chat_completion"
    ):
        errors.append("runner_receipt_incomplete")
    if list(evidence.get("signals_sent", [])):
        errors.append("unexpected_signal")
    metrics = _row_metrics(proposal_rows, forced_prefix_rows)
    for model_id in REQUIRED_MODEL_IDS:
        if metrics["empty_output_rate_by_model"][model_id] >= 0.05:
            errors.append(f"empty_output_rate_exceeded:{model_id}")
        if metrics["zero_token_rate_by_model"][model_id] > 0:
            errors.append(f"zero_token_output:{model_id}")
        if metrics["parseable_rate_by_model"][model_id] < 0.90:
            errors.append(f"parseable_rate_below_floor:{model_id}")
        if metrics["leaked_control_token_count_by_model"][model_id] > 0:
            errors.append(f"leaked_control_token:{model_id}")
        if metrics["length_limited_count_by_model"][model_id] > 0:
            errors.append(f"length_limited_output:{model_id}")
    return list(dict.fromkeys(errors))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the terminal artifact except its self-referential digest."""

    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_text(canonical_json(stable))


def collect_preconditions(
    *,
    chat_canary_path: Path = CHAT_CANARY_PATH,
    expected_chat_canary_hash: str = PINNED_CHAT_CANARY_SHA256,
    chat_canary_validator: Callable[[Mapping[str, Any]], list[str]] = chat_canary.validate_artifact,
    base_collector: Callable[..., JsonDict] = chat_canary.collect_preconditions,
    fixture_path: Path,
    expected_fixture_hash: str,
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
    **base_kwargs: Any,
) -> JsonDict:
    """Add the approved chat canary to the full local-resource preflight."""

    canary: JsonDict = {}
    canary_hash = None
    canary_errors = []
    try:
        canary = json.loads(chat_canary_path.read_text(encoding="utf-8"))
        canary_hash = sha256_file(chat_canary_path)
        canary_errors = chat_canary_validator(canary)
    except Exception as exc:  # noqa: BLE001 - exact read failures belong in the artifact.
        canary = {"read_error": f"{type(exc).__name__}: {exc}"}
    base = base_collector(
        fixture_path=fixture_path,
        expected_fixture_hash=expected_fixture_hash,
        model_specs=model_specs,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        **base_kwargs,
    )
    chat_checks = [
        gate_row(
            "chat_transport_source_hash",
            expected_chat_canary_hash,
            canary_hash,
            canary_hash == expected_chat_canary_hash,
        ),
        gate_row(
            "chat_transport_ready_score",
            1,
            canary.get("chat_transport_ready_score"),
            canary.get("chat_transport_ready_score") == 1 and not canary_errors,
        ),
    ]
    checks = [*chat_checks, *deepcopy(list(base.get("checks", [])))]
    return {
        **deepcopy(base),
        "all_passed": all(row.get("passed") is True for row in checks),
        "checks": checks,
        "chat_canary": canary,
        "chat_canary_hash": canary_hash,
        "chat_canary_validation_errors": canary_errors,
        "upstream_gate_rows": [
            *deepcopy(chat_checks),
            *deepcopy(list(base.get("upstream_gate_rows", []))),
        ],
    }


def phase_evidence(
    phase_rows: Sequence[Mapping[str, Any]], preconditions: Mapping[str, Any]
) -> JsonDict:
    """Project model, lease, checkpoint, cleanup, and GPU phase receipts."""

    executions = [
        {
            "model_id": row.get("model_id"),
            "phase": row.get("phase"),
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
            "phase": row.get("phase"),
            "path": row.get("checkpoint_path"),
            "sha256": row.get("checkpoint_sha256"),
            "manifest_hash": row.get("manifest_hash"),
            "row_count": len(row.get("raw_rows", [])),
        }
        for row in phase_rows
    ]
    cleanups = []
    for row in phase_rows:
        cleanup = deepcopy(dict(row.get("cleanup") or {}))
        cleanup["model_id"] = row.get("model_id")
        cleanup["phase"] = row.get("phase")
        cleanups.append(cleanup)
    releases = [
        {
            "model_id": row.get("model_id"),
            "phase": row.get("phase"),
            **deepcopy(dict(row.get("vram_release") or {})),
        }
        for row in phase_rows
    ]
    stage_samples = [
        {
            "model_id": row.get("model_id"),
            "phase": row.get("phase"),
            **deepcopy(dict(sample)),
        }
        for row in phase_rows
        for sample in row.get("gpu_sample_rows", [])
    ]
    task_samples = [
        {
            "model_id": row.get("model_id"),
            "phase": row.get("phase"),
            **deepcopy(dict(sample)),
        }
        for row in phase_rows
        for sample in row.get("task_gpu_samples", [])
    ]
    peaks: dict[str, int] = {}
    for sample in task_samples:
        for device in sample.get("devices", []):
            uuid = str(device.get("uuid"))
            peaks[uuid] = max(peaks.get(uuid, 0), int(device.get("memory_used_mb", 0) or 0))
    signals = [signal for row in cleanups for signal in row.get("signals_sent", [])]
    return {
        "model_identity_rows": deepcopy(list(preconditions.get("model_identity_rows", []))),
        "model_execution_rows": executions,
        "checkpoint_rows": checkpoints,
        "gpu_lease_rows": prior_bank.normalize_gpu_lease_rows(phase_rows),
        "vram_release_rows": releases,
        "cleanup_rows": cleanups,
        "runner_receipt": deepcopy(dict(preconditions.get("runner_receipt") or {})),
        "signals_sent": signals,
        "stage_gpu_telemetry_rows": stage_samples,
        "task_gpu_telemetry_rows": task_samples,
        "peak_vram_by_device": peaks,
    }


def _all_models_real(
    model_specs: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    execution_rows: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(
        [row.get("hf_id") for row in model_specs] == list(REQUIRED_MODEL_IDS)
        and not model_spec_errors(model_specs)
        and not model_identity_errors(model_specs, identity_rows)
        and {
            (str(row.get("model_id")), str(row.get("phase")))
            for row in execution_rows
            if int(row.get("offloaded_layers", 0) or 0) > 0 and row.get("used_both_gpus") is True
        }
        == _expected_shards()
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    proposal_rows: Sequence[Mapping[str, Any]] = (),
    forced_prefix_rows: Sequence[Mapping[str, Any]] = (),
    ordered_unit_ids: Sequence[str] = (),
    forced_unit_ids: Sequence[str] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    phase_order_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one schema-complete blocked, partial, null, or positive artifact."""

    proposals = [deepcopy(dict(row)) for row in proposal_rows]
    forced = [deepcopy(dict(row)) for row in forced_prefix_rows]
    facts = deepcopy(dict(evidence or {}))
    identity_rows = deepcopy(
        list(facts.get("model_identity_rows", preconditions.get("model_identity_rows", [])))
    )
    facts["model_identity_rows"] = identity_rows
    views = proposal_evidence_rows(proposals)
    all_rows = [*proposals, *forced]
    metrics = _row_metrics(proposals, forced)
    unit_rows = list(dict(preconditions.get("fixture") or {}).get("unit_rows", []))
    preflight = preconditions.get("all_passed") is True
    errors = (
        completion_errors(
            proposal_rows=proposals,
            forced_prefix_rows=forced,
            ordered_unit_ids=ordered_unit_ids,
            forced_unit_ids=forced_unit_ids,
            unit_rows=unit_rows,
            model_specs=model_specs,
            identity_rows=identity_rows,
            evidence=facts,
        )
        if preflight
        else ["preconditions_failed"]
    )
    complete = bool(preflight and not errors)
    expected_row_count = len(REQUIRED_MODEL_IDS) * (
        len(ordered_unit_ids) * len(PROPOSAL_SEEDS) + len(forced_unit_ids)
    )
    if not preflight:
        verdict_class = "blocked"
        verdict = "blocked: a required precondition failed before model generation"
    elif complete:
        verdict_class = "positive"
        verdict = "positive: complete three-family chat entrance bank acquired"
    elif len(all_rows) == expected_row_count:
        verdict_class = "null"
        verdict = "null: complete generation evidence failed a transport gate"
    else:
        verdict_class = "partial"
        verdict = "partial: the launched run did not acquire every scheduled row"
    gate_rows = [
        gate_row("model_family_count", 3, len(model_specs), len(model_specs) == 3),
        gate_row(
            "proposal_row_completeness",
            [],
            [error for error in errors if "proposal" in error or "fixture" in error],
            not any("proposal" in error or "fixture" in error for error in errors),
        ),
        gate_row(
            "forced_prefix_row_completeness",
            [],
            [error for error in errors if "forced" in error],
            not any("forced" in error for error in errors),
        ),
        gate_row(
            "transport_telemetry_cleanup",
            [],
            [
                error
                for error in errors
                if "proposal" not in error and "fixture" not in error and "forced" not in error
            ],
            not any(
                "proposal" not in error and "fixture" not in error and "forced" not in error
                for error in errors
            ),
        ),
    ]
    execution_rows = deepcopy(list(facts.get("model_execution_rows", [])))
    model_loads = {
        phase: sum(
            int(row.get("model_load_count", 0) or 0)
            for row in execution_rows
            if row.get("phase") == phase
        )
        for phase in ("proposal", "forced_prefix")
    }
    model_durations = {
        model_id: sum(
            float(row.get("duration_s", 0.0) or 0.0)
            for row in execution_rows
            if row.get("model_id") == model_id
        )
        for model_id in REQUIRED_MODEL_IDS
    }
    exact_by_model = prior_bank._per_model_results(proposals, forced)
    transport_by_model = {row["model_id"]: row for row in metrics["per_model_transport_rows"]}
    per_model_rows = [{**row, **transport_by_model[row["model_id"]]} for row in exact_by_model]
    token_score_rows = deepcopy(views["token_score_rows"])
    for forced_row in forced:
        for token_index, token_score in enumerate(forced_row.get("token_scores", [])):
            token_score_rows.append(
                {
                    "raw_key": forced_row.get("raw_key"),
                    "raw_output_hash": forced_row.get("raw_output_hash"),
                    "model_id": forced_row.get("model_id"),
                    "unit_id": forced_row.get("unit_id"),
                    "seed": forced_row.get("seed"),
                    "arm": "forced_prefix",
                    "token_index": token_index,
                    **deepcopy(dict(token_score)),
                }
            )
    source_hashes = deepcopy(dict(source_artifact_hashes or {}))
    real = _all_models_real(model_specs, identity_rows, execution_rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "model_full_generation" if preflight else "blocked_no_run",
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": source_hashes,
        "cited_upstream_artifacts": [
            {
                "path": str(FIXTURE_PATH.relative_to(REPO_ROOT)),
                "sha256": preconditions.get("fixture_hash"),
                "fields_imported": ["entrance_fixture_ready_score", "unit_rows", "entrance_rows"],
            },
            {
                "path": str(LEASE_AUDIT_PATH.relative_to(REPO_ROOT)),
                "sha256": preconditions.get("lease_audit_hash"),
                "fields_imported": ["gpu_lease_cold_audit_ready_score"],
            },
            {
                "path": str(PRIOR_BANK_PATH.relative_to(REPO_ROOT)),
                "sha256": source_hashes.get(PRIOR_BANK_PATH.name),
                "fields_imported": ["prior_incomplete_run"],
            },
            {
                "path": str(CHAT_CANARY_PATH.relative_to(REPO_ROOT)),
                "sha256": preconditions.get("chat_canary_hash"),
                "fields_imported": ["chat_transport_ready_score", "sampling_config"],
            },
        ],
        "upstream_gate_rows": deepcopy(list(preconditions.get("upstream_gate_rows", []))),
        "rows": gate_rows,
        "per_game_results": prior_bank._per_game_results(proposals, forced),
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
            for row in all_rows
        ],
        "rendered_prompt_hash_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "rendered_prompt_hash": row.get("rendered_prompt_hash"),
                "available": row.get("rendered_prompt_hash_available"),
            }
            for row in all_rows
        ],
        "stop_config_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "stop": deepcopy(row.get("stop_config")),
            }
            for row in all_rows
        ],
        "runner_build_rows": deepcopy(list(preconditions.get("runner_build_rows", []))),
        "proposal_rows": proposals,
        "raw_proposal_rows": views["raw_proposal_rows"],
        "token_score_rows": token_score_rows,
        "token_count_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "prefix_token_count": row.get("prefix_token_count"),
                "effective_completion_budget_tokens": row.get("effective_completion_budget_tokens"),
            }
            for row in all_rows
        ],
        "finish_reason_rows": [
            {
                "raw_key": row.get("raw_key"),
                "model_id": row.get("model_id"),
                "finish_reason": row.get("finish_reason"),
            }
            for row in all_rows
        ],
        "parse_rows": [
            *deepcopy(views["parse_rows"]),
            *[
                {
                    "raw_key": row.get("raw_key"),
                    "raw_output_hash": row.get("raw_output_hash"),
                    "model_id": row.get("model_id"),
                    "unit_id": row.get("unit_id"),
                    "seed": row.get("seed"),
                    "arm": "forced_prefix",
                    "parse_failure": row.get("parse_failure"),
                }
                for row in forced
            ],
        ],
        "exact_label_rows": views["exact_label_rows"],
        "causal_witness_rows": views["causal_witness_rows"],
        "guess_without_witness_rows": views["guess_without_witness_rows"],
        "forced_prefix_rows": forced,
        "per_model_rows": per_model_rows,
        "per_source_group_rows": prior_bank._per_source_group_results(
            proposals,
            forced,
            unit_rows,
        ),
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_hash": sha256_text(PROMPT_TEMPLATE),
        "sampling_config": deepcopy(GENERATION_CONFIG),
        "seed_rows": [
            {"arm": "proposal", "seeds": list(PROPOSAL_SEEDS)},
            {"arm": "forced_prefix", "seeds": [FORCED_PREFIX_SEED]},
            {"purpose": "phase_order", "seed": RANDOM_SEED},
        ],
        "phase_order_rows": [deepcopy(dict(row)) for row in phase_order_rows],
        "ordered_unit_ids": list(ordered_unit_ids),
        "forced_unit_ids": list(forced_unit_ids),
        "checkpoint_rows": deepcopy(list(facts.get("checkpoint_rows", []))),
        "runner_receipt": deepcopy(dict(facts.get("runner_receipt") or {})),
        "generation_invoked": bool(all_rows),
        "total_model_count": len({str(row.get("model_id")) for row in all_rows}),
        "model_load_count_by_stage": model_loads,
        "per_model_duration_s": model_durations,
        "stage_gpu_telemetry_rows": deepcopy(list(facts.get("stage_gpu_telemetry_rows", []))),
        "task_gpu_telemetry_rows": deepcopy(list(facts.get("task_gpu_telemetry_rows", []))),
        "peak_vram_by_device": deepcopy(dict(facts.get("peak_vram_by_device", {}))),
        "gpu_topology_rows": deepcopy(
            list(dict(preconditions.get("gpu_topology") or {}).get("devices", []))
        ),
        "gpu_lease_rows": deepcopy(list(facts.get("gpu_lease_rows", []))),
        "vram_release_rows": deepcopy(list(facts.get("vram_release_rows", []))),
        "cleanup_rows": deepcopy(list(facts.get("cleanup_rows", []))),
        "model_file_hash_rows": deepcopy(list(facts.get("model_file_hash_rows", []))),
        "signals_sent": deepcopy(list(facts.get("signals_sent", []))),
        "empty_output_rate_by_model": metrics["empty_output_rate_by_model"],
        "zero_token_rate_by_model": metrics["zero_token_rate_by_model"],
        "parseable_rate_by_model": metrics["parseable_rate_by_model"],
        "leaked_control_token_count_by_model": metrics["leaked_control_token_count_by_model"],
        "length_limited_count_by_model": metrics["length_limited_count_by_model"],
        "all_models_real": real,
        "entrance_proposal_bank_complete_score": int(complete and real),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": (
            gate_summary(preconditions.get("checks", []))
            if not preflight
            else gate_summary(gate_rows)
        ),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check raw projections, aggregates, terminal state, and checksum."""

    errors = [
        f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact
    ]
    if errors:
        return errors
    if set(REQUIRED_ARTIFACT_FIELDS) != set(artifact.get("field_principles", {})) or any(
        not str(value).strip() for value in artifact.get("field_principles", {}).values()
    ):
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
    views = proposal_evidence_rows(proposals)
    for field in (
        "raw_proposal_rows",
        "exact_label_rows",
        "causal_witness_rows",
        "guess_without_witness_rows",
    ):
        if artifact.get(field) != views[field]:
            errors.append(f"{field.removesuffix('_rows')}_projection_mismatch")
    if any(
        field in row
        for row in artifact.get("raw_proposal_rows", [])
        for field in ("legal", "reachable", "duplicate", "causal_witness", "guess_without_witness")
    ):
        errors.append("raw_row_contains_exact_label")
    identity_rows = list(artifact.get("model_identity_rows", []))
    execution_rows = list(artifact.get("model_execution_rows", []))
    evidence = {
        "model_identity_rows": identity_rows,
        "model_execution_rows": execution_rows,
        "checkpoint_rows": list(artifact.get("checkpoint_rows", [])),
        "gpu_lease_rows": list(artifact.get("gpu_lease_rows", [])),
        "vram_release_rows": list(artifact.get("vram_release_rows", [])),
        "cleanup_rows": list(artifact.get("cleanup_rows", [])),
        "model_file_hash_rows": list(artifact.get("model_file_hash_rows", [])),
        "stage_gpu_telemetry_rows": list(artifact.get("stage_gpu_telemetry_rows", [])),
        "task_gpu_telemetry_rows": list(artifact.get("task_gpu_telemetry_rows", [])),
        "peak_vram_by_device": dict(artifact.get("peak_vram_by_device", {})),
        "runner_receipt": dict(artifact.get("runner_receipt") or {}),
        "signals_sent": list(artifact.get("signals_sent", [])),
    }
    preflight = dict(artifact.get("preconditions_checked") or {}).get("all_passed") is True
    completion = (
        completion_errors(
            proposal_rows=proposals,
            forced_prefix_rows=forced,
            ordered_unit_ids=list(artifact.get("ordered_unit_ids", [])),
            forced_unit_ids=list(artifact.get("forced_unit_ids", [])),
            unit_rows=list(
                dict(artifact.get("preconditions_checked") or {})
                .get("fixture", {})
                .get("unit_rows", [])
            ),
            model_specs=list(artifact.get("model_specs", [])),
            identity_rows=identity_rows,
            evidence=evidence,
        )
        if preflight
        else ["preconditions_failed"]
    )
    real = _all_models_real(list(artifact.get("model_specs", [])), identity_rows, execution_rows)
    expected_score = int(preflight and not completion and real)
    if score != expected_score:
        errors.append("completion_score_mismatch")
    if artifact.get("all_models_real") is not real:
        errors.append("all_models_real_mismatch")
    metrics = _row_metrics(proposals, forced)
    for name in (
        "empty_output_rate_by_model",
        "zero_token_rate_by_model",
        "parseable_rate_by_model",
        "leaked_control_token_count_by_model",
        "length_limited_count_by_model",
    ):
        if artifact.get(name) != metrics[name]:
            errors.append(f"{name}_mismatch")
    all_rows = [*proposals, *forced]
    if artifact.get("generation_invoked") is not bool(all_rows):
        errors.append("generation_invoked_mismatch")
    if artifact.get("total_model_count") != len({str(row.get("model_id")) for row in all_rows}):
        errors.append("total_model_count_mismatch")
    expected_class = "model_full_generation" if preflight else "blocked_no_run"
    if artifact.get("inference_substrate_class") != expected_class:
        errors.append("inference_substrate_class_mismatch")
    expected_loads = {
        phase: sum(
            int(row.get("model_load_count", 0) or 0)
            for row in execution_rows
            if row.get("phase") == phase
        )
        for phase in ("proposal", "forced_prefix")
    }
    if artifact.get("model_load_count_by_stage") != expected_loads:
        errors.append("model_load_count_projection_mismatch")
    expected_durations = {
        model_id: sum(
            float(row.get("duration_s", 0.0) or 0.0)
            for row in execution_rows
            if row.get("model_id") == model_id
        )
        for model_id in REQUIRED_MODEL_IDS
    }
    if artifact.get("per_model_duration_s") != expected_durations:
        errors.append("per_model_duration_projection_mismatch")
    verdict_class = str(artifact.get("verdict_class", ""))
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not verdict.startswith(f"{verdict_class}:"):
        errors.append("honest_verdict_prefix_mismatch")
    expected_rows = len(REQUIRED_MODEL_IDS) * (
        len(artifact.get("ordered_unit_ids", [])) * len(PROPOSAL_SEEDS)
        + len(artifact.get("forced_unit_ids", []))
    )
    if not preflight:
        summary = dict(artifact.get("gate_check_summary") or {})
        if verdict_class != "blocked" or score != 0:
            errors.append("blocked_verdict_mismatch")
        if not all(key in summary for key in ("failed_check", "expected_value", "observed_value")):
            errors.append("blocked_gate_summary_incomplete")
    elif expected_score:
        if verdict_class != "positive":
            errors.append("positive_verdict_mismatch")
    elif len(all_rows) == expected_rows:
        if verdict_class != "null":
            errors.append("null_verdict_mismatch")
    elif verdict_class != "partial":
        errors.append("partial_verdict_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _source_hashes() -> JsonDict:  # pragma: no cover - hashes the live repository state.
    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-references.md"),
        Path("openspec/capabilities/llm-ebm-inference/spec.md"),
        Path("openspec/capabilities/verification/spec.md"),
        Path("python/carnot/inference/sota_models.py"),
        Path("python/carnot/experiment_7080_v620_three_family_entrance_bank.py"),
        Path("python/carnot/experiment_7085_v621_chat_transport_canary.py"),
        Path("python/carnot/experiment_7086_v621_three_family_entrance_bank.py"),
        Path("tests/python/test_experiment_7086_v621_three_family_entrance_bank.py"),
        Path("scripts/experiments/experiment_7086_v621_three_family_entrance_bank.py"),
        Path("scripts/run_stop_authority.py"),
        FIXTURE_PATH.relative_to(REPO_ROOT),
        LEASE_AUDIT_PATH.relative_to(REPO_ROOT),
        PRIOR_BANK_PATH.relative_to(REPO_ROOT),
        CHAT_CANARY_PATH.relative_to(REPO_ROOT),
    )
    rows = [
        {
            "path": str(path),
            "exists": (REPO_ROOT / path).is_file(),
            "sha256": sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None,
        }
        for path in paths
    ]
    return {
        "files": rows,
        "by_name": {Path(row["path"]).name: row["sha256"] for row in rows},
        "all_present": all(row["exists"] for row in rows),
        "manifest_hash": sha256_text(canonical_json(rows)),
    }


def run_model_phase(  # pragma: no cover - live CUDA subprocess boundary.
    *,
    model: Mapping[str, Any],
    phase: str,
    schedule_rows: Sequence[Mapping[str, Any]],
    devices: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    checkpoint_dir: Path,
) -> JsonDict:
    """Reuse the proven owned-worker boundary with this experiment's worker."""

    saved = (
        chat_canary.MODULE_NAME,
        chat_canary.EXPERIMENT_ID,
        chat_canary.GENERATION_CONFIG,
        chat_canary.LEASE_RUNTIME_DIR,
        chat_canary.MODEL_TIMEOUT_S,
    )
    chat_canary.MODULE_NAME = MODULE_NAME
    chat_canary.EXPERIMENT_ID = EXPERIMENT_ID
    chat_canary.GENERATION_CONFIG = deepcopy(GENERATION_CONFIG)
    chat_canary.LEASE_RUNTIME_DIR = LEASE_RUNTIME_DIR
    chat_canary.MODEL_TIMEOUT_S = MODEL_TIMEOUT_S
    try:
        row = chat_canary.run_model_phase(
            model=model,
            schedule_rows=schedule_rows,
            devices=devices,
            raw_dir=raw_dir / phase,
            checkpoint_dir=checkpoint_dir / phase,
        )
    finally:
        (
            chat_canary.MODULE_NAME,
            chat_canary.EXPERIMENT_ID,
            chat_canary.GENERATION_CONFIG,
            chat_canary.LEASE_RUNTIME_DIR,
            chat_canary.MODEL_TIMEOUT_S,
        ) = saved
    row["phase"] = phase
    if isinstance(row.get("cleanup"), dict):
        row["cleanup"]["phase"] = phase
    return row


def _worker_main(
    payload_path: Path, output_path: Path, ready_path: Path, port: int
) -> int:  # pragma: no cover - private worker process boundary.
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
        write_json_atomic(output_path, worker_run_schedule(payload))
        return 0
    except Exception as exc:  # noqa: BLE001 - parent retains the exact failure.
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


def run(  # pragma: no cover - required six-shard live experiment.
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    fixture_path: Path = FIXTURE_PATH,
    raw_dir: Path = RAW_DIR,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Preflight, acquire raw shards, label, validate, and write once."""

    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or resolve_model_specs())]
    preconditions = collect_preconditions(
        fixture_path=fixture_path,
        expected_fixture_hash=PINNED_FIXTURE_SHA256,
        model_specs=specs,
        result_path=result_path,
        checkpoint_path=checkpoint_dir / "write-probe.json",
    )
    source_manifest = _source_hashes()
    source_hashes = dict(source_manifest["by_name"])
    source_hashes["manifest_hash"] = source_manifest["manifest_hash"]
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
    fixture = dict(preconditions["fixture"])
    ordered_unit_ids = list(fixture["split_manifest"]["ordered_unit_ids"])
    units_by_id = {str(row["unit_id"]): row for row in fixture["unit_rows"]}
    visible_by_id = {str(row["unit_id"]): row for row in fixture["model_visible_rows"]}
    visible_units = [visible_by_id[unit_id] for unit_id in ordered_unit_ids]
    proposal_schedule = build_proposal_schedule(specs, visible_units, ordered_unit_ids)
    schedule_errors = matched_schedule_errors(proposal_schedule, ordered_unit_ids)
    if schedule_errors:
        raise RuntimeError(f"proposal_schedule_invalid:{schedule_errors}")
    phase_order = randomized_phase_order()
    devices = list(preconditions["gpu_topology"]["devices"])
    phases = []
    for order in [row for row in phase_order if row["phase"] == "proposal"]:
        model = next(row for row in specs if row["hf_id"] == order["model_id"])
        phase = run_model_phase(
            model=model,
            phase="proposal",
            schedule_rows=[row for row in proposal_schedule if row["model_id"] == model["hf_id"]],
            devices=devices,
            raw_dir=raw_dir,
            checkpoint_dir=checkpoint_dir,
        )
        phases.append(phase)
        if phase.get("terminal_state") != "complete":
            break
    proposal_raw = [row for phase in phases for row in phase.get("raw_rows", [])]
    proposal_raw.sort(key=lambda row: str(row.get("raw_key")))
    proposals = label_proposal_rows(proposal_raw, fixture["entrance_rows"])
    forced_unit_ids = select_diversity_unit_ids(
        [units_by_id[unit_id] for unit_id in ordered_unit_ids]
    )
    prefixes = select_forced_prefixes(
        REQUIRED_MODEL_IDS,
        fixture["unit_rows"],
        fixture["entrance_rows"],
        proposals,
        forced_unit_ids,
    )
    forced_schedule = build_forced_schedule(prefixes, specs)
    if (
        len(proposal_raw) == len(REQUIRED_MODEL_IDS) * 96 * len(PROPOSAL_SEEDS)
        and len(prefixes) == 72
    ):
        for order in [row for row in phase_order if row["phase"] == "forced_prefix"]:
            model = next(row for row in specs if row["hf_id"] == order["model_id"])
            phase = run_model_phase(
                model=model,
                phase="forced_prefix",
                schedule_rows=[row for row in forced_schedule if row["model_id"] == model["hf_id"]],
                devices=devices,
                raw_dir=raw_dir,
                checkpoint_dir=checkpoint_dir,
            )
            phases.append(phase)
            if phase.get("terminal_state") != "complete":
                break
    forced_raw = [
        row
        for phase in phases
        if phase.get("phase") == "forced_prefix"
        for row in phase.get("raw_rows", [])
    ]
    forced_raw.sort(key=lambda row: str(row.get("raw_key")))
    forced = label_forced_rows(forced_raw, fixture["unit_rows"])
    evidence = phase_evidence(phases, preconditions)
    evidence["model_file_hash_rows"] = [
        {
            "model_id": row["hf_id"],
            "path": row["model_path"],
            "sha256": sha256_file(row["model_path"]),
        }
        for row in specs
    ]
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions=preconditions,
        proposal_rows=proposals,
        forced_prefix_rows=forced,
        ordered_unit_ids=ordered_unit_ids,
        forced_unit_ids=forced_unit_ids,
        source_artifact_hashes=source_hashes,
        evidence=evidence,
        phase_order_rows=phase_order,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command surface.
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
            args.worker_payload,
            args.worker_output,
            args.worker_ready,
            args.worker_port,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
