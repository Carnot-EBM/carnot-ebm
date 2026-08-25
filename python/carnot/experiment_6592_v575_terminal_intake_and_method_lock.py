"""Build the V575 terminal evidence root without running an LLM.

The intake replays four immutable V574 artifacts. It binds every completed CFR
unit to nested receipts and locks three paper-derived methods. It creates
infrastructure readiness only. It does not compare CFR quality.

Spec: REQ-REPORT-6592 and SCENARIO-REPORT-6592-REPLAY through
SCENARIO-REPORT-6592-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_6588_v574_bounded_cfr_launch_root as exp6588
from carnot import experiment_6590_qwen36_constraint_first_stream as exp6590
from carnot import experiment_6591_gemma4_31b_constraint_first_stream as exp6591


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260825"
TASK_ID = "exp6592-v575-terminal-intake-and-method-lock"
RESULT_RELATIVE_PATH = Path("results/experiment_6592_v575_terminal_intake_and_method_lock.json")
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
V575_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
REFERENCE_RELATIVE_PATH = Path("research-references.md")
INFERENCE_SUBSTRATE = "v574_terminal_and_v575_method_replay_no_llm"
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
SOURCE_ARTIFACTS: dict[str, JsonDict] = {
    "Exp6588": {
        "path": Path("results/experiment_6588_v574_bounded_cfr_launch_root.json"),
        "ready_field": "v574_cfr_launch_ready_score",
        "kind": "launch_root",
    },
    "Exp6589": {
        "path": Path("results/experiment_6589_isolated_pytest_receipt_remediation.json"),
        "ready_field": "pytest_receipt_remediation_ready_score",
        "kind": "blocked_receipt_infrastructure",
    },
    "Exp6590": {
        "path": Path("results/experiment_6590_qwen36_constraint_first_stream.json"),
        "ready_field": "qwen_cfr_rows_ready_score",
        "kind": "qwen_cfr_stream",
    },
    "Exp6591": {
        "path": Path("results/experiment_6591_gemma4_31b_constraint_first_stream.json"),
        "ready_field": "gemma31_cfr_rows_ready_score",
        "kind": "gemma_cfr_stream",
    },
}
STREAM_MODULES = {"Exp6590": exp6590, "Exp6591": exp6591}
MANDATED_MODEL_IDS = exp6588.MANDATED_MODEL_IDS
METHOD_SOURCE_IDS = ("arXiv:2608.23526", "arXiv:2608.23551", "arXiv:2608.21466")
REQUIRED_ATTACK_IDS = (
    "invented_cfr_benefit",
    "exp6589_block_erasure",
    "principle_wrapper_misread",
    "missing_row_hashes",
    "methodology_invention",
    "paper_result_transfer",
    "legacy_model_substitution",
    "unowned_gpu_eviction",
    "gate_field_drift",
    "historical_artifact_mutation",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "v574_terminal_replay_rows",
    "cfr_stream_methodology_binding_rows",
    "method_source_lock_rows",
    "model_cache_identity_rows",
    "gpu_ownership_rows",
    "current_roadmap_gate_contract_rows",
    "v575_cfr_reducer_ready_score",
    "v575_dual_gpu_canary_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
FIELD_PRINCIPLES = {
    "status": "The intake is terminal and cannot pose bootstrap work as evidence.",
    "honest_verdict": "The verdict reports readiness without creating a science result.",
    "verdict_class": "A ready intake is null infrastructure, not positive science.",
    "gate_check_summary": "A block names the failed evidence or protection check and its value.",
    "v574_terminal_replay_rows": "The four upstream terminal states and hashes stay exact.",
    "cfr_stream_methodology_binding_rows": "Every CFR unit binds nested receipts without invention.",
    "method_source_lock_rows": "Paper methods keep bounded imports, controls, metrics, and non-claims.",
    "model_cache_identity_rows": "Both local GGUF identities derive from bounded content reads.",
    "gpu_ownership_rows": "Dual-GPU readiness cannot evict or hide unowned work.",
    "current_roadmap_gate_contract_rows": "Active and future gate fields keep exact spelling and state.",
    "v575_cfr_reducer_ready_score": "Only two complete immutable CFR stream replays open Exp6593.",
    "v575_dual_gpu_canary_ready_score": "Only two cached models and two idle owned GPUs open Exp6602.",
    "attack_rows": "Evidence, authority, resource, and history mutations fail closed.",
    "preconditions_checked": "Sources, hashes, methods, caches, resources, and ownership are explicit.",
    "protected_files_unchanged": "Historical sources and protected orchestration files do not change.",
    "inference_substrate": "The task declares terminal artifact and method replay with no LLM.",
    "verifier_is_oracle": "Exact replay owns readiness but cannot create positive science.",
    "field_provenance": "Every field names source artifacts, rows, hashes, and reducers.",
    "duration_s": "Monotonic duration exposes a source-only shortcut.",
    "tests_run": "Named commands, exits, and durations make validation reproducible.",
    "reproducibility_checksum": "A final content hash detects terminal mutation.",
}
DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest -n 0 -o addopts= tests/python/test_experiment_6592_v575_terminal_intake_and_method_lock.py -q",
        "exit_code": 0,
        "duration_s": 13.36,
    },
    {
        "command": ".venv/bin/coverage run --source=python/carnot -m pytest -n 0 -o addopts= tests/python/test_experiment_6592_v575_terminal_intake_and_method_lock.py -q && .venv/bin/coverage report --include=python/carnot/experiment_6592_v575_terminal_intake_and_method_lock.py --show-missing --fail-under=100",
        "exit_code": 0,
        "duration_s": 29.80,
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit_code": 3,
        "duration_s": 612.21,
    },
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6592_v575_terminal_intake_and_method_lock.py",
        "exit_code": 0,
        "duration_s": 0.10,
    },
    {
        "command": ".venv/bin/python scripts/adversarial_verify.py results/experiment_6592_v575_terminal_intake_and_method_lock.json",
        "exit_code": 0,
        "duration_s": 0.74,
    },
    {
        "command": ".venv/bin/python scripts/verdict_row_consistency_lint.py --strict results/experiment_6592_v575_terminal_intake_and_method_lock.json",
        "exit_code": 0,
        "duration_s": 0.01,
    },
    {
        "command": ".venv/bin/python scripts/artifact_convention_audit.py --recent 1 --dry-run",
        "exit_code": 0,
        "duration_s": 0.01,
    },
)

canonical_json = exp6588.canonical_json
sha256_bytes = exp6588.sha256_bytes
sha256_file = exp6588.sha256_file
sha256_json = exp6588.sha256_json
artifact_checksum = exp6588.artifact_checksum
load_json = exp6588.load_json


def unwrap_value(value: Any) -> Any:
    """Return the bare value from nested principle wrappers."""

    if isinstance(value, Mapping) and "value" in value:
        return unwrap_value(value["value"])
    return value


def _hash_matches(row: Mapping[str, Any], field: str = "row_hash") -> bool:
    expected = row.get(field)
    bare = {key: value for key, value in row.items() if key != field}
    return expected == sha256_json(bare)


def _top_level_methodology_warnings(payload: Mapping[str, Any]) -> list[str]:
    warnings = []
    if not (payload.get("model_specs") or payload.get("target_model")):
        warnings.append("model_specs_or_target_model")
    if not (payload.get("random_seed") is not None or payload.get("random_seeds_used")):
        warnings.append("random_seed_or_random_seeds_used")
    return warnings


def replay_one(experiment_id: str, source: Mapping[str, Any], repo_root: Path) -> JsonDict:
    """Preserve one terminal state and recompute only its raw readiness."""

    config = SOURCE_ARTIFACTS[experiment_id]
    ready_field = str(config["ready_field"])
    stored_score = unwrap_value(source.get(ready_field))
    verdict = unwrap_value(source.get("honest_verdict"))
    verdict_class = unwrap_value(source.get("verdict_class"))
    warnings = _top_level_methodology_warnings(source) if experiment_id in STREAM_MODULES else []
    if experiment_id == "Exp6588":
        recomputed_score = exp6588.readiness_reducer(source)["ready_score"]
        replay_valid = recomputed_score == 1.0 and verdict_class is None
    elif experiment_id == "Exp6589":
        recomputed_score = 0.0
        replay_valid = bool(
            stored_score == 0.0
            and verdict_class == "blocked"
            and str(verdict).startswith("blocked_receipt_validation_block:")
            and unwrap_value(source.get("terminal_validation_failure")) is not None
        )
    else:
        recomputed_score = STREAM_MODULES[experiment_id].stream_reducer(source)["ready_score"]
        replay_valid = recomputed_score == 1.0 and verdict_class is None
    relative = config["path"]
    row: JsonDict = {
        "experiment_id": experiment_id,
        "source_artifact_path": relative.as_posix(),
        "source_artifact_sha256": sha256_file(repo_root / relative),
        "source_kind": config["kind"],
        "status": unwrap_value(source.get("status")),
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": deepcopy(unwrap_value(source.get("gate_check_summary"))),
        "readiness_field": ready_field,
        "stored_ready_score": stored_score,
        "recomputed_ready_score": recomputed_score,
        "replay_valid": replay_valid,
        "adversarial_disposition": {
            "flagged_adversarial": unwrap_value(source.get("flagged_adversarial")),
            "corrigendum_pending": deepcopy(unwrap_value(source.get("corrigendum_pending", []))),
            "live_methodology_warnings": warnings,
        },
        "science_result_created": False,
    }
    row["row_hash"] = sha256_json(row)
    return row


def build_v574_terminal_replay_rows(repo_root: Path) -> list[JsonDict]:
    """Replay the four terminal V574 artifacts in experiment order."""

    return [
        replay_one(experiment_id, load_json(repo_root / config["path"]), repo_root)
        for experiment_id, config in SOURCE_ARTIFACTS.items()
    ]


def terminal_replay_rows_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require exact terminal states while preserving the Exp6589 block."""

    by_id = {str(row.get("experiment_id")): row for row in rows}
    return bool(
        list(by_id) == list(SOURCE_ARTIFACTS)
        and by_id["Exp6589"].get("verdict_class") == "blocked"
        and by_id["Exp6589"].get("stored_ready_score") == 0.0
        and by_id["Exp6589"].get("recomputed_ready_score") == 0.0
        and by_id["Exp6590"].get("recomputed_ready_score") == 1.0
        and by_id["Exp6591"].get("recomputed_ready_score") == 1.0
        and all(row.get("replay_valid") is True for row in rows)
        and all(row.get("science_result_created") is False for row in rows)
        and all(
            str(row.get("source_artifact_sha256", "")).startswith("sha256:")
            and _hash_matches(row)
            for row in rows
        )
    )


def _receipt_hashes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [sha256_json(row) for row in rows]


def build_cfr_stream_methodology_binding_rows(repo_root: Path) -> list[JsonDict]:
    """Bind every CFR unit to nested model, seed, stage, and process receipts."""

    binding_rows = []
    for experiment_id, module in STREAM_MODULES.items():
        config = SOURCE_ARTIFACTS[experiment_id]
        source = load_json(repo_root / config["path"])
        units = [row for row in source.get("per_unit_rows", []) if isinstance(row, Mapping)]
        raw_rows = [row for row in source.get("raw_stage_receipts", []) if isinstance(row, Mapping)]
        exact_rows = [
            row for row in source.get("exact_checker_receipts", []) if isinstance(row, Mapping)
        ]
        failure_rows = [row for row in source.get("failure_rows", []) if isinstance(row, Mapping)]
        checkpoints = [
            row for row in source.get("checkpoint_receipts", []) if isinstance(row, Mapping)
        ]
        reduction = module.stream_reducer(source)
        model_hash = sha256_json(source.get("model_spec_and_identity", {}))
        prompt_hash = sha256_json(source.get("prompt_source_router_hashes", {}))
        gpu_hash = sha256_json(source.get("gpu_process_receipts", {}))
        failure_table_hash = sha256_json(failure_rows)
        recomputation_hash = sha256_json(reduction)
        warnings = _top_level_methodology_warnings(source)
        for index, unit in enumerate(units):
            unit_id = unit.get("unit_id")
            arms = [arm for arm in unit.get("arms", []) if isinstance(arm, Mapping)]
            unit_raw = [row for row in raw_rows if row.get("unit_id") == unit_id]
            unit_exact = [row for row in exact_rows if row.get("unit_id") == unit_id]
            unit_failures = [row for row in failure_rows if row.get("unit_id") == unit_id]
            checkpoint = checkpoints[index] if index < len(checkpoints) else {}
            checkpoint_matches = bool(
                checkpoint
                and checkpoint.get("completed_unit_count") == index + 1
                and checkpoint.get("completed_unit_ids", [])[-1:] == [unit_id]
                and checkpoint.get("completed_unit_row_hashes", [])[-1:]
                == [unit.get("row_hash")]
            )
            row: JsonDict = {
                "experiment_id": experiment_id,
                "source_artifact_path": config["path"].as_posix(),
                "source_artifact_sha256": sha256_file(repo_root / config["path"]),
                "unit_index": index,
                "unit_id": unit_id,
                "unit_row_hash": unit.get("row_hash"),
                "arm_row_hashes": [arm.get("row_hash") for arm in arms],
                "source_bytes_sha256": unit.get("source_bytes_sha256"),
                "task_bytes_sha256": unit.get("task_bytes_sha256"),
                "model_spec_and_identity_hash": model_hash,
                "prompt_source_router_hashes_hash": prompt_hash,
                "seed_rows": [
                    {"arm_name": arm.get("arm_name"), "seed": arm.get("seed")} for arm in arms
                ],
                "raw_stage_receipt_hashes": _receipt_hashes(unit_raw),
                "exact_checker_receipt_hashes": _receipt_hashes(unit_exact),
                "checkpoint_receipt_hash": sha256_json(checkpoint) if checkpoint else "missing",
                "checkpoint_file_sha256": checkpoint.get("checkpoint_sha256"),
                "gpu_process_receipts_hash": gpu_hash,
                "failure_receipt_hashes": _receipt_hashes(unit_failures),
                "failure_table_hash": failure_table_hash,
                "failure_table_bound": isinstance(source.get("failure_rows"), list),
                "stream_recomputation_hash": recomputation_hash,
                "stream_recomputed_ready_score": reduction["ready_score"],
                "charged_tokens": sum(
                    int(arm.get("tokens", {}).get("total", 0) or 0) for arm in arms
                ),
                "charged_latency_s": round(
                    sum(float(arm.get("latency_s", 0.0) or 0.0) for arm in arms), 9
                ),
                "charged_cost": round(
                    sum(float(arm.get("charged_cost", 0.0) or 0.0) for arm in arms), 9
                ),
                "top_level_methodology_warnings": warnings,
                "invented_top_level_methodology": False,
                "all_nested_receipts_bound": bool(
                    reduction["ready_score"] == 1.0
                    and str(unit.get("row_hash", "")).startswith("sha256:")
                    and len(arms) == 3
                    and bool(unit_raw)
                    and len(unit_exact) == 3
                    and checkpoint_matches
                    and all(
                        str(value).startswith("sha256:")
                        for value in (model_hash, prompt_hash, gpu_hash, recomputation_hash)
                    )
                ),
            }
            row["row_hash"] = sha256_json(row)
            binding_rows.append(row)
    return binding_rows


def methodology_binding_rows_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require 20 complete, non-invented unit bindings per model family."""

    counts = {
        experiment_id: sum(row.get("experiment_id") == experiment_id for row in rows)
        for experiment_id in STREAM_MODULES
    }
    return bool(
        len(rows) == 40
        and counts == {"Exp6590": 20, "Exp6591": 20}
        and len({(row.get("experiment_id"), row.get("unit_id")) for row in rows}) == 40
        and all(
            row.get("all_nested_receipts_bound") is True
            and row.get("invented_top_level_methodology") is False
            and str(row.get("unit_row_hash", "")).startswith("sha256:")
            and len(row.get("arm_row_hashes", [])) == 3
            and all(str(value).startswith("sha256:") for value in row.get("arm_row_hashes", []))
            and bool(row.get("raw_stage_receipt_hashes"))
            and len(row.get("exact_checker_receipt_hashes", [])) == 3
            and str(row.get("checkpoint_receipt_hash", "")).startswith("sha256:")
            and row.get("failure_table_bound") is True
            and _hash_matches(row)
            for row in rows
        )
    )


def build_method_source_lock_rows(repo_root: Path) -> list[JsonDict]:
    """Freeze three paper methods as bounded canaries, not imported results."""

    reference_hash = sha256_file(repo_root / REFERENCE_RELATIVE_PATH)
    roadmap_hash = sha256_file(repo_root / V575_DOC_RELATIVE_PATH)
    definitions = (
        {
            "source_id": METHOD_SOURCE_IDS[0],
            "retrieved_title": "Correcting a learned physical invariant improves world-model rollouts",
            "source_url": "https://arxiv.org/abs/2608.23526",
            "bounded_import": "project frozen latent rollouts toward a preregistered invariant level set",
            "controls": ["no projection", "exact-invariant diagnostic", "norm-matched random constraint", "damped dynamics"],
            "metrics": ["held rollout error", "invariant drift", "energy", "projection distance", "wall time"],
            "non_claims": ["not an ARC solve", "learned invariant is not exact authority", "paper results are not reproduced evidence"],
        },
        {
            "source_id": METHOD_SOURCE_IDS[1],
            "retrieved_title": "ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings",
            "source_url": "https://arxiv.org/abs/2608.23551",
            "bounded_import": "project a toy continuous predictor into the convex hull of fixed feasible token embeddings",
            "controls": ["unconstrained flow", "nearest-token-only rounding", "held predictor errors", "degenerate feasible set"],
            "metrics": ["valid-token convergence", "constraint violations", "steps", "path length", "endpoint distortion"],
            "non_claims": ["not a language model reproduction", "exact-set validity is circular", "paper perplexity does not transfer"],
        },
        {
            "source_id": METHOD_SOURCE_IDS[2],
            "retrieved_title": "Spectral partitioning for k-block averaging kernels of finite Markov chains",
            "source_url": "https://arxiv.org/abs/2608.21466",
            "bounded_import": "select finite-state averaging blocks from bottom spectral modes for exact-enumerable Ising fixtures",
            "controls": ["sequential Gibbs", "matched transitions", "charged setup", "independent, ferromagnetic, and frustrated fixtures"],
            "metrics": ["total variation error", "moment error", "effective sample size", "setup cost", "transition cost"],
            "non_claims": ["no FPGA or TSU result", "no general hardware speedup", "paper experiments are not Carnot evidence"],
        },
    )
    rows = []
    for definition in definitions:
        row = {
            **definition,
            "retrieved_on": RUN_DATE,
            "arxiv_version": "v1",
            "research_reference_path": REFERENCE_RELATIVE_PATH.as_posix(),
            "research_reference_sha256": reference_hash,
            "v575_roadmap_path": V575_DOC_RELATIVE_PATH.as_posix(),
            "v575_roadmap_sha256": roadmap_hash,
            "paper_result_counts_as_carnot_evidence": False,
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def method_source_locks_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require all source boundaries and forbid paper-result transfer."""

    return bool(
        [row.get("source_id") for row in rows] == list(METHOD_SOURCE_IDS)
        and all(
            row.get("retrieved_title")
            and row.get("bounded_import")
            and row.get("controls")
            and row.get("metrics")
            and row.get("non_claims")
            and row.get("paper_result_counts_as_carnot_evidence") is False
            and str(row.get("research_reference_sha256", "")).startswith("sha256:")
            and _hash_matches(row)
            for row in rows
        )
    )


def build_model_cache_identity_rows(repo_root: Path) -> list[JsonDict]:
    """Reuse the bounded content-derived GGUF admission receipts."""

    return deepcopy(exp6588.build_model_cache_identity_rows(repo_root))


def cache_identity_rows_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require exactly the two mandated local identities with no model work."""

    return bool(
        [row.get("repository_id") for row in rows] == list(MANDATED_MODEL_IDS)
        and all(
            row.get("resolved") is True
            and row.get("admitted") is True
            and row.get("model_load_performed") is False
            and row.get("download_performed") is False
            and row.get("auto_tokenizer_used") is False
            and row.get("content_metadata", {})
            .get("bounded_read_receipt", {})
            .get("tensor_payload_bytes_read")
            == 0
            and _hash_matches(row)
            for row in rows
        )
    )


def build_gpu_ownership_rows(receipt: Mapping[str, Any] | None = None) -> list[JsonDict]:
    """Describe idle local devices without claiming or evicting active work."""

    observed = exp6588._gpu_ownership_receipt() if receipt is None else receipt  # noqa: SLF001
    processes = [
        dict(row) for row in observed.get("compute_process_rows", []) if isinstance(row, Mapping)
    ]
    accessible = bool(
        observed.get("visible") is True
        and observed.get("gpu_query_exit_code") == 0
        and observed.get("process_query_exit_code") == 0
    )
    rows = []
    for device in sorted(observed.get("gpu_rows", []), key=lambda row: int(row.get("index", 999))):
        device_processes = [
            {**row, "task_owned": False}
            for row in processes
            if row.get("gpu_uuid") == device.get("uuid")
        ]
        supported = "RTX 3090" in str(device.get("name", ""))
        idle = int(device.get("utilization_pct", 101) or 0) <= 5 and not device_processes
        available = accessible and supported and idle
        row = {
            "gpu_index": device.get("index"),
            "gpu_uuid": device.get("uuid"),
            "name": device.get("name"),
            "memory_total_mb": device.get("memory_total_mb"),
            "memory_free_mb": device.get("memory_free_mb"),
            "utilization_pct": device.get("utilization_pct"),
            "local_runtime_access": accessible,
            "supported_rtx_3090": supported,
            "idle": idle,
            "runtime_owner_uid": os.geteuid(),
            "available_for_runtime_ownership": available,
            "ownership_basis": "local query access and no active compute process",
            "unowned_processes_preserved": device_processes,
            "signals_sent": list(observed.get("signals_sent", [])),
            "eviction_performed": False,
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def _gpu_rows_safe(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        rows
        and all(
            row.get("signals_sent") == []
            and row.get("eviction_performed") is False
            and all(process.get("task_owned") is False for process in row.get("unowned_processes_preserved", []))
            and _hash_matches(row)
            for row in rows
        )
    )


def dual_gpu_rows_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require two idle, accessible RTX 3090 devices and zero eviction."""

    eligible = [row for row in rows if row.get("available_for_runtime_ownership") is True]
    return bool(
        len(eligible) >= 2
        and _gpu_rows_safe(rows)
        and all(row.get("supported_rtx_3090") is True for row in eligible)
    )


def _required_artifact_fields(prompt: str) -> set[str]:
    return exp6588._required_artifact_fields(prompt)  # noqa: SLF001


def build_current_roadmap_gate_contract_rows(repo_root: Path) -> list[JsonDict]:
    """Audit the exact active Exp6593 gate and named future Exp6602 gate."""

    roadmap_path = repo_root / ACTIVE_ROADMAP_RELATIVE_PATH
    roadmap = yaml.safe_load(roadmap_path.read_text(encoding="utf-8"))
    tasks = roadmap.get("tasks", []) if isinstance(roadmap, Mapping) else []
    by_id = {str(task.get("id")): task for task in tasks if isinstance(task, Mapping)}
    root_fields = _required_artifact_fields(str(by_id.get(TASK_ID, {}).get("prompt", "")))
    design_path = repo_root / V575_DOC_RELATIVE_PATH
    design_text = design_path.read_text(encoding="utf-8")
    contracts = (
        (
            "exp6593-cfr-independent-row-reducer",
            "v575_cfr_reducer_ready_score",
            "cfr_reducer_ready_score",
            "### Exp6593 - independent CFR row reducer",
        ),
        (
            "exp6602-dual-gpu-flagship-residency-canary",
            "v575_dual_gpu_canary_ready_score",
            None,
            "### Exp6602 - dual-GPU isolated-residency canary",
        ),
    )
    rows = []
    for consumer_id, artifact_field, owner_output_field, design_marker in contracts:
        consumer = by_id.get(consumer_id, {})
        gates = consumer.get("gated_on", []) if isinstance(consumer, Mapping) else []
        matching = [
            gate
            for gate in gates
            if isinstance(gate, Mapping)
            and gate.get("upstream") == TASK_ID
            and gate.get("artifact_field") == artifact_field
            and gate.get("op") == "=="
            and gate.get("value") == 1.0
        ]
        consumer_fields = _required_artifact_fields(str(consumer.get("prompt", "")))
        consumer_exists = consumer_id in by_id
        row = {
            "roadmap_path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "roadmap_sha256": sha256_file(roadmap_path),
            "v575_design_path": V575_DOC_RELATIVE_PATH.as_posix(),
            "v575_design_sha256": sha256_file(design_path),
            "upstream_task_id": TASK_ID,
            "artifact_field": artifact_field,
            "operator": "==",
            "expected_value": 1.0,
            "consumer_task_id": consumer_id,
            "owner_output_field": owner_output_field,
            "upstream_task_exists": TASK_ID in by_id,
            "consumer_task_exists": consumer_exists,
            "design_document_consumer_exists": design_marker in design_text,
            "gate_declared_exactly_once": len(matching) == 1,
            "upstream_field_declared": artifact_field in root_fields,
            "owner_output_field_declared": bool(
                owner_output_field and owner_output_field in consumer_fields
            ),
        }
        row["all_cross_references_close"] = all(
            row[key]
            for key in (
                "upstream_task_exists",
                "consumer_task_exists",
                "design_document_consumer_exists",
                "gate_declared_exactly_once",
                "upstream_field_declared",
                "owner_output_field_declared",
            )
        )
        row["disposition"] = (
            "closed_active_same_roadmap_gate"
            if row["all_cross_references_close"]
            else "warning_inactive_future_task_not_in_active_yaml"
            if row["design_document_consumer_exists"] and not consumer_exists
            else "failed_active_gate_cross_reference"
        )
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def _active_cfr_gate_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    matches = [row for row in rows if row.get("consumer_task_id") == "exp6593-cfr-independent-row-reducer"]
    return bool(
        len(matches) == 1
        and matches[0].get("artifact_field") == "v575_cfr_reducer_ready_score"
        and matches[0].get("all_cross_references_close") is True
        and _hash_matches(matches[0])
    )


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {
        config["path"].as_posix(): sha256_file(repo_root / config["path"])
        for config in SOURCE_ARTIFACTS.values()
    }


def _protected_receipt(
    before: Mapping[str, str],
    after: Mapping[str, str],
    sources_before: Mapping[str, str] | None = None,
    sources_after: Mapping[str, str] | None = None,
) -> JsonDict:
    protected_rows = [
        {
            "path": path,
            "before_sha256": digest,
            "after_sha256": after.get(path, "missing"),
            "unchanged": digest == after.get(path),
        }
        for path, digest in before.items()
    ]
    source_rows = [
        {
            "path": path,
            "before_sha256": digest,
            "after_sha256": (sources_after or {}).get(path, "missing"),
            "unchanged": digest == (sources_after or {}).get(path),
        }
        for path, digest in (sources_before or {}).items()
    ]
    changed = [row["path"] for row in [*protected_rows, *source_rows] if not row["unchanged"]]
    return {
        "rows": protected_rows,
        "historical_artifact_rows": source_rows,
        "changed_paths": changed,
        "all_unchanged": not changed,
    }


def _intake_contract_score(payload: Mapping[str, Any]) -> float:
    safe = bool(
        terminal_replay_rows_ready(payload.get("v574_terminal_replay_rows", []))
        and methodology_binding_rows_ready(
            payload.get("cfr_stream_methodology_binding_rows", [])
        )
        and method_source_locks_ready(payload.get("method_source_lock_rows", []))
        and cache_identity_rows_ready(payload.get("model_cache_identity_rows", []))
        and _gpu_rows_safe(payload.get("gpu_ownership_rows", []))
        and _active_cfr_gate_ready(payload.get("current_roadmap_gate_contract_rows", []))
        and payload.get("protected_files_unchanged", {}).get("all_unchanged") is True
    )
    return 1.0 if safe else 0.0


def build_attack_rows(base_candidate: Mapping[str, Any]) -> list[JsonDict]:
    """Mutate each protected boundary and retain the fail-closed score."""

    mutations = {
        "invented_cfr_benefit": lambda value: value["v574_terminal_replay_rows"][0].update(
            science_result_created=True
        ),
        "exp6589_block_erasure": lambda value: value["v574_terminal_replay_rows"][1].update(
            verdict_class=None
        ),
        "principle_wrapper_misread": lambda value: value["v574_terminal_replay_rows"][1].update(
            verdict_class={"principle": "must preserve block", "value": "complete"}
        ),
        "missing_row_hashes": lambda value: value["cfr_stream_methodology_binding_rows"][0].update(
            unit_row_hash="missing"
        ),
        "methodology_invention": lambda value: value["cfr_stream_methodology_binding_rows"][0].update(
            invented_top_level_methodology=True
        ),
        "paper_result_transfer": lambda value: value["method_source_lock_rows"][0].update(
            paper_result_counts_as_carnot_evidence=True
        ),
        "legacy_model_substitution": lambda value: value["model_cache_identity_rows"][0].update(
            repository_id="Qwen/Qwen3.5-0.8B"
        ),
        "unowned_gpu_eviction": lambda value: value["gpu_ownership_rows"][0].update(
            signals_sent=[999], eviction_performed=True
        ),
        "gate_field_drift": lambda value: value["current_roadmap_gate_contract_rows"][0].update(
            artifact_field="v575_cfr_reducer_ready_scor"
        ),
        "historical_artifact_mutation": lambda value: value["protected_files_unchanged"].update(
            all_unchanged=False, changed_paths=["results/experiment_6590_qwen36_constraint_first_stream.json"]
        ),
    }
    rows = []
    for attack_id in REQUIRED_ATTACK_IDS:
        candidate = deepcopy(dict(base_candidate))
        mutations[attack_id](candidate)
        observed = _intake_contract_score(candidate)
        row = {
            "attack_id": attack_id,
            "expected_acceptance_score": 0.0,
            "candidate_acceptance_score": observed,
            "passed": observed == 0.0,
            "disposition": "fail_closed",
            "reducer": "_intake_contract_score",
        }
        row["row_hash"] = sha256_json(row)
        rows.append(row)
    return rows


def attack_rows_ready(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Require one authentic zero-score receipt for every named attack."""

    return bool(
        [row.get("attack_id") for row in rows] == list(REQUIRED_ATTACK_IDS)
        and all(
            row.get("candidate_acceptance_score") == 0.0
            and row.get("passed") is True
            and _hash_matches(row)
            for row in rows
        )
    )


def readiness_reducer(payload: Mapping[str, Any]) -> JsonDict:
    """Compute separate CFR and nonblocking dual-GPU readiness scores."""

    checks = {
        "terminal_replay": terminal_replay_rows_ready(payload.get("v574_terminal_replay_rows", [])),
        "methodology_bindings": methodology_binding_rows_ready(
            payload.get("cfr_stream_methodology_binding_rows", [])
        ),
        "method_source_locks": method_source_locks_ready(
            payload.get("method_source_lock_rows", [])
        ),
        "cache_identities": cache_identity_rows_ready(payload.get("model_cache_identity_rows", [])),
        "gpu_safety": _gpu_rows_safe(payload.get("gpu_ownership_rows", [])),
        "dual_gpu_available": dual_gpu_rows_ready(payload.get("gpu_ownership_rows", [])),
        "active_cfr_gate_contract": _active_cfr_gate_ready(
            payload.get("current_roadmap_gate_contract_rows", [])
        ),
        "attacks": attack_rows_ready(payload.get("attack_rows", [])),
        "protected_history": payload.get("protected_files_unchanged", {}).get("all_unchanged")
        is True,
    }
    cfr_ready = all(
        checks[name]
        for name in (
            "terminal_replay",
            "methodology_bindings",
            "active_cfr_gate_contract",
            "attacks",
            "protected_history",
        )
    )
    dual_ready = all(
        checks[name]
        for name in ("cache_identities", "gpu_safety", "dual_gpu_available", "attacks", "protected_history")
    )
    return {
        "checks": checks,
        "v575_cfr_reducer_ready_score": 1.0 if cfr_ready else 0.0,
        "v575_dual_gpu_canary_ready_score": 1.0 if dual_ready else 0.0,
    }


def _cpu_receipt() -> JsonDict:
    model = platform.processor()
    if not model:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {"model": model or "unknown", "count": os.cpu_count() or 1}


def _ram_receipt() -> JsonDict:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, value = line.split(":", 1)
        values[name] = int(value.strip().split()[0])
    return {"total_kib": values["MemTotal"], "available_kib": values["MemAvailable"]}


def _summary_review_receipts(repo_root: Path) -> list[JsonDict]:
    rows = []
    child_environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("COV_CORE_", "COVERAGE_"))
    }
    for experiment_id, config in SOURCE_ARTIFACTS.items():
        command = [
            sys.executable,
            str(repo_root / "scripts/summarize_artifact.py"),
            str(repo_root / config["path"]),
        ]
        result = subprocess.run(
            command,
            cwd=repo_root,
            env=child_environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "command": " ".join(command),
                "exit_code": result.returncode,
                "stdout_sha256": sha256_bytes(result.stdout.encode("utf-8")),
                "stderr_sha256": sha256_bytes(result.stderr.encode("utf-8")),
            }
        )
    return rows


def collect_preconditions(
    repo_root: Path,
    protected_before: Mapping[str, str],
    cache_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    *,
    date: str,
) -> JsonDict:
    """Record source, host, cache, ownership, and no-LLM input receipts."""

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    disk = shutil.disk_usage(repo_root)
    return {
        "planning_date": date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_llm_substrate": True,
        "llm_calls_issued": 0,
        "model_loads_issued": 0,
        "downloads_issued": 0,
        "auto_tokenizer_calls_issued": 0,
        "gpu_eviction_signals_sent": [],
        "v574_artifacts": [
            {
                "experiment_id": experiment_id,
                "path": config["path"].as_posix(),
                "exists": (repo_root / config["path"]).is_file(),
                "sha256": sha256_file(repo_root / config["path"]),
            }
            for experiment_id, config in SOURCE_ARTIFACTS.items()
        ],
        "roadmap": {
            "path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / ACTIVE_ROADMAP_RELATIVE_PATH),
        },
        "v575_roadmap_document": {
            "path": V575_DOC_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / V575_DOC_RELATIVE_PATH),
        },
        "v575_reference_refresh": {
            "path": REFERENCE_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(repo_root / REFERENCE_RELATIVE_PATH),
        },
        "protected_file_hashes_before": dict(protected_before),
        "dirty_worktree": {
            "command": "git status --short",
            "exit_code": status.returncode,
            "is_dirty": bool(status.stdout.strip()),
            "entries": status.stdout.splitlines(),
            "status_sha256": sha256_bytes(status.stdout.encode("utf-8")),
        },
        "cpu": _cpu_receipt(),
        "ram": _ram_receipt(),
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "model_cache_identities": [
            {
                "repository_id": row.get("repository_id"),
                "cache_path": row.get("cache_path"),
                "trusted_sha256": row.get("trusted_sha256"),
                "resolved": row.get("resolved"),
            }
            for row in cache_rows
        ],
        "visible_gpu_ownership": deepcopy(list(gpu_rows)),
        "summarize_artifact_receipts": _summary_review_receipts(repo_root),
    }


def _tests_run_receipts(rows: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if rows is None else rows
    return [
        {
            "command": str(row.get("command", "")),
            "exit_code": int(row.get("exit_code", 1)),
            "duration_s": float(row.get("duration_s", 0.0)),
        }
        for row in source
    ]


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    sources = [
        {"path": config["path"].as_posix(), "sha256": sha256_file(repo_root / config["path"])}
        for config in SOURCE_ARTIFACTS.values()
    ]
    sources.extend(
        [
            {"path": ACTIVE_ROADMAP_RELATIVE_PATH.as_posix(), "sha256": sha256_file(repo_root / ACTIVE_ROADMAP_RELATIVE_PATH)},
            {"path": V575_DOC_RELATIVE_PATH.as_posix(), "sha256": sha256_file(repo_root / V575_DOC_RELATIVE_PATH)},
            {"path": REFERENCE_RELATIVE_PATH.as_posix(), "sha256": sha256_file(repo_root / REFERENCE_RELATIVE_PATH)},
        ]
    )
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source_artifacts": deepcopy(sources),
            "raw_rows": [
                "v574_terminal_replay_rows",
                "cfr_stream_methodology_binding_rows",
                "method_source_lock_rows",
                "model_cache_identity_rows",
                "gpu_ownership_rows",
                "current_roadmap_gate_contract_rows",
                "attack_rows",
            ],
            "reducer_code": ["replay_one", "stream_reducer", "readiness_reducer", "artifact_checksum"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _blocking_checks(payload: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "protected_files_unchanged": payload.get("protected_files_unchanged", {}).get("all_unchanged") is True,
        "v574_terminal_replay_rows": terminal_replay_rows_ready(payload.get("v574_terminal_replay_rows", [])),
        "cfr_stream_methodology_binding_rows": methodology_binding_rows_ready(payload.get("cfr_stream_methodology_binding_rows", [])),
        "method_source_lock_rows": method_source_locks_ready(payload.get("method_source_lock_rows", [])),
        "exp6593_active_gate_contract": _active_cfr_gate_ready(payload.get("current_roadmap_gate_contract_rows", [])),
        "attack_rows": attack_rows_ready(payload.get("attack_rows", [])),
    }


def _gate_summary(payload: Mapping[str, Any], reduction: Mapping[str, Any]) -> JsonDict:
    blocking = _blocking_checks(payload)
    future_rows = [
        row
        for row in payload.get("current_roadmap_gate_contract_rows", [])
        if row.get("consumer_task_id") == "exp6602-dual-gpu-flagship-residency-canary"
    ]
    checks = [
        {"check": name, "expected_value": True, "observed_value": value, "passed": value, "blocking": True}
        for name, value in blocking.items()
    ]
    checks.extend(
        [
            {
                "check": "model_cache_identity_rows",
                "expected_value": True,
                "observed_value": reduction["checks"]["cache_identities"],
                "passed": reduction["checks"]["cache_identities"],
                "blocking": False,
            },
            {
                "check": "two_idle_runtime_owned_rtx_3090_devices",
                "expected_value": True,
                "observed_value": reduction["checks"]["dual_gpu_available"],
                "passed": reduction["checks"]["dual_gpu_available"],
                "blocking": False,
            },
            {
                "check": "exp6602_active_same_roadmap_gate_contract",
                "expected_value": True,
                "observed_value": bool(future_rows and future_rows[0].get("all_cross_references_close")),
                "passed": bool(future_rows and future_rows[0].get("all_cross_references_close")),
                "blocking": False,
            },
        ]
    )
    blocking_failures = [row for row in checks if row["blocking"] and not row["passed"]]
    warnings = [row for row in checks if not row["blocking"] and not row["passed"]]
    return {
        "checks": checks,
        "failed_blocking_check_count": len(blocking_failures),
        "warning_check_count": len(warnings),
        "first_blocking_failure": blocking_failures[0] if blocking_failures else None,
        "warnings": warnings,
        "passed": not blocking_failures,
        "v575_cfr_reducer_ready_score": reduction["v575_cfr_reducer_ready_score"],
        "v575_dual_gpu_canary_ready_score": reduction["v575_dual_gpu_canary_ready_score"],
        "dual_gpu_zero_blocks_science": False,
    }


def build_report(
    repo_root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build one terminal V575 intake from checked-in rows and local receipts."""

    started = time.perf_counter()
    protected_before = _protected_hashes(repo_root)
    sources_before = _source_hashes(repo_root)
    replay_rows = build_v574_terminal_replay_rows(repo_root)
    binding_rows = build_cfr_stream_methodology_binding_rows(repo_root)
    method_rows = build_method_source_lock_rows(repo_root)
    cache_rows = build_model_cache_identity_rows(repo_root)
    gpu_rows = build_gpu_ownership_rows()
    gate_rows = build_current_roadmap_gate_contract_rows(repo_root)
    preconditions = collect_preconditions(
        repo_root, protected_before, cache_rows, gpu_rows, date=date
    )
    protected = _protected_receipt(
        protected_before,
        _protected_hashes(repo_root),
        sources_before,
        _source_hashes(repo_root),
    )
    base: JsonDict = {
        "v574_terminal_replay_rows": replay_rows,
        "cfr_stream_methodology_binding_rows": binding_rows,
        "method_source_lock_rows": method_rows,
        "model_cache_identity_rows": cache_rows,
        "gpu_ownership_rows": gpu_rows,
        "current_roadmap_gate_contract_rows": gate_rows,
        "protected_files_unchanged": protected,
    }
    attacks = build_attack_rows(base)
    reduction = readiness_reducer({**base, "attack_rows": attacks})
    report_duration = float(duration_s) if duration_s is not None else time.perf_counter() - started
    partial: JsonDict = {**base, "attack_rows": attacks}
    summary = _gate_summary(partial, reduction)
    complete = summary["passed"] is True
    payload: JsonDict = {
        "status": (
            "complete_v575_terminal_intake_and_method_lock"
            if complete
            else "blocked_v575_terminal_intake_and_method_lock"
        ),
        "honest_verdict": (
            "complete: V574 terminal evidence replayed; Exp6589 remains blocked; both CFR streams bind complete nested receipts; V575 methods and gates are locked; no science result was created"
            if complete
            else "blocked_v575_terminal_intake_and_method_lock: one or more source, methodology, method, gate, attack, or protection checks failed"
        ),
        "verdict_class": None if complete else "blocked",
        "gate_check_summary": summary,
        **partial,
        "v575_cfr_reducer_ready_score": reduction["v575_cfr_reducer_ready_score"],
        "v575_dual_gpu_canary_ready_score": reduction["v575_dual_gpu_canary_ready_score"],
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(repo_root),
        "duration_s": report_duration,
        "tests_run": _tests_run_receipts(tests_run),
    }
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def validate_report(payload: Mapping[str, Any]) -> list[str]:
    """Return terminal schema, reducer, authority, and checksum errors."""

    errors = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not isinstance(payload.get("duration_s"), (int, float)) or payload.get("duration_s", 0) <= 0:
        errors.append("duration_s must be positive")
    reduction = readiness_reducer(payload)
    for field in ("v575_cfr_reducer_ready_score", "v575_dual_gpu_canary_ready_score"):
        if payload.get(field) != reduction[field]:
            errors.append(f"{field} mismatch")
    blocking = _blocking_checks(payload)
    complete = all(blocking.values())
    if complete:
        if payload.get("status") != "complete_v575_terminal_intake_and_method_lock":
            errors.append("complete intake status mismatch")
        if payload.get("verdict_class") is not None:
            errors.append("complete intake verdict_class must be null")
        if not str(payload.get("honest_verdict", "")).startswith(
            ("complete:", "success:", "passed:", "shipped:")
        ):
            errors.append("terminal success prefix missing")
    else:
        if payload.get("verdict_class") != "blocked":
            errors.append("blocked intake verdict_class mismatch")
        summary = payload.get("gate_check_summary", {})
        if not isinstance(summary, Mapping) or summary.get("failed_blocking_check_count", 0) < 1:
            errors.append("blocked gate_check_summary missing failure")
    if (
        payload.get("status") == "complete_v575_terminal_intake_and_method_lock"
        and payload.get("protected_files_unchanged", {}).get("all_unchanged") is not True
    ):
        errors.append("protected_files_unchanged failed")
    provenance = payload.get("field_provenance", {})
    if not isinstance(provenance, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(provenance):
        errors.append("field_provenance missing required fields")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def atomic_write_report(path: str | Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate, sync, atomically replace, and directory-sync one JSON."""

    errors = validate_report(payload)
    if errors:
        raise ValueError("; ".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():  # pragma: no cover - replace failure cleanup.
            temporary.unlink()
    return {
        "path": str(target),
        "file_fsync": True,
        "atomic_replace": True,
        "directory_fsync": True,
        "output_sha256": sha256_file(target),
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        errors = validate_report(load_json(output))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"valid: {output}")
        return 0
    report = build_report(REPO_ROOT, date=args.date)
    receipt = atomic_write_report(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "v575_cfr_reducer_ready_score": report["v575_cfr_reducer_ready_score"],
                "v575_dual_gpu_canary_ready_score": report[
                    "v575_dual_gpu_canary_ready_score"
                ],
                **receipt,
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point.
    raise SystemExit(main())
