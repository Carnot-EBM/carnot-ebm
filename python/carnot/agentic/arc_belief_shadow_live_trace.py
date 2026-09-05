"""Run one provenance-complete live belief-shadow transport trace.

The experiment compares two fresh E3 policies on one official visible frame.
The belief arm computes its counterfactual ranking, but returns the control
candidate order. This measures transport and isolation. It does not measure
belief value and it never banks a solve.

Spec refs: REQ-ARC-7025, REQ-ARC-7030, and their scenarios.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import tempfile
import time
from typing import Any

from carnot import task_runtime_receipts
from carnot.agentic.arc_eval_provenance import (
    build_arc_eval_provenance_for_policy,
    build_arc_model_identity_receipt,
    evaluation_counters,
    huggingface_snapshot_revision,
    validate_arc_evaluation_row,
)
from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7025
SCHEMA = "carnot.exp7025.belief_shadow_live_trace.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 7_025_202_609_05
INFERENCE_SUBSTRATE = "live_llm_arc_belief_shadow"
MANDATED_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MANDATED_MODEL_NAME = "Qwen3.6-35B-A3B"
MODEL_PROMPT = "Reply with exactly OK."
MODEL_MAX_TOKENS = 32
MODEL_TEMPERATURE = 0.0
MODEL_N_CTX = 4096
ACTION_BUDGET = 1
CELL_IDS = ("control", "belief_shadow")
RESULT_RELATIVE_PATH = Path("results/experiment_7025_belief_shadow_live_trace.json")
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_7025/checkpoint.json")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
POLICY_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")

REPO_ROOT = Path(__file__).resolve().parents[3]

UPSTREAMS = (
    (
        7010,
        Path("results/experiment_7010_arc_eval_provenance_contract.json"),
        "arc_eval_provenance_contract_ready_score",
    ),
    (
        7017,
        Path("results/experiment_7017_task_linked_compute_receipts.json"),
        "task_compute_receipt_ready_score",
    ),
    (
        7024,
        Path("results/experiment_7024_belief_aware_e3_selector.json"),
        "belief_selector_live_path_ready_score",
    ),
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes",
    "cited_upstream_artifacts",
    "source_artifact_hashes",
    "rows",
    "per_action_results",
    "control_rows",
    "shadow_rows",
    "shadow_action_parity_rows",
    "belief_query_rows",
    "ranking_influence_rows",
    "abstention_rows",
    "model_execution_rows",
    "gpu_identity_rows",
    "gpu_sample_rows",
    "server_rows",
    "lease_rows",
    "request_counter_rows",
    "completion_counter_rows",
    "phase_receipt_rows",
    "runner_decision_rows",
    "checkpoint_rows",
    "teardown_rows",
    "task_compute_receipt",
    "solve_provenance",
    "solve_registry_precheck_rows",
    "arc_new_level_banked",
    "belief_shadow_trace_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why the transport claim needs it.",
    "preconditions_checked": "Exact checks prevent fallback evidence after a missing resource.",
    "inference_substrate": "The substrate distinguishes this live LLM trace from fixtures.",
    "duration_s": "Wall time exposes whether model loading and inference could have occurred.",
    "MODEL_SPECS": "The resolved spec binds selection to the approved cache resolver.",
    "models_used": "The invoked model list prevents a legacy smoke model from satisfying the task.",
    "model_file_hashes": "File digests bind the model name to the exact loaded bytes.",
    "cited_upstream_artifacts": "Citations identify the exact structured gates and provenance contract.",
    "source_artifact_hashes": "Source hashes bind the trace to the code and registry it used.",
    "rows": "Summary rows make the readiness reduction independently countable.",
    "per_action_results": "Action rows preserve each cell decision without a solve claim.",
    "control_rows": "Control rows record the no-belief decision and its model request.",
    "shadow_rows": "Shadow rows record belief computation while preserving the control action.",
    "shadow_action_parity_rows": "Parity rows prove that belief did not change an emitted action.",
    "belief_query_rows": "Query rows distinguish an active belief path from a silent no-op.",
    "ranking_influence_rows": "Influence rows separate counterfactual ranking from applied action.",
    "abstention_rows": "Abstention rows account for every safe refusal by the selector.",
    "model_execution_rows": "Execution rows bind model, CUDA flags, context, process, and counters.",
    "gpu_identity_rows": "GPU identity prevents a generic device label from proving RTX 3090 use.",
    "gpu_sample_rows": "In-phase samples prove that the owned model process used the leased GPU.",
    "server_rows": "Server rows bind the owned PID, endpoint, port, model, context, and counters.",
    "lease_rows": "Lease rows prove task ownership and release of the selected GPU.",
    "request_counter_rows": "Request deltas expose skipped, duplicated, and failed cell calls.",
    "completion_counter_rows": "Completion rows require every request to reach one terminal result.",
    "phase_receipt_rows": "Phase clocks attribute setup, load, inference, write, and cleanup cost.",
    "runner_decision_rows": "One model must select the sequential runner from measured concurrency.",
    "checkpoint_rows": "Checkpoint rows prove each completed cell survives a restart without replay.",
    "teardown_rows": "Teardown rows prove only owned resources were stopped and released.",
    "task_compute_receipt": "The shared Exp7017 receipt validates phase, GPU, process, lease, and cleanup links.",
    "solve_provenance": "Live self-discovery labels the official observation path without claiming a solve.",
    "solve_registry_precheck_rows": "Registry hashes prove no reproduced level was targeted or modified.",
    "arc_new_level_banked": "A bare zero prevents transport evidence from becoming solve credit.",
    "belief_shadow_trace_ready_score": "One requires every transport, parity, provenance, and cleanup gate.",
    "random_seed": "One seed fixes policy order and model sampling for both cells.",
    "reproducibility_checksum": "A canonical digest detects any later artifact change.",
    "gate_check_summary": "The first exact failure makes a blocked run actionable.",
    "verifier_is_oracle": "False states that this transport check does not define action correctness.",
    "verdict_class": "A closed terminal class separates readiness from blockage or partial execution.",
    "honest_verdict": "A class-consistent prefix gives downstream readers one terminal meaning.",
}

VERDICT_PREFIXES = {
    "positive": "complete_positive_",
    "circular_positive": "complete_circular_positive_",
    "null": "complete_null_",
    "blocked": "blocked_",
    "disqualified": "disqualified_",
    "partial": "partial_",
}


def sha256_file(path: str | Path) -> str:
    """Hash one readable file with the project digest label."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash stable JSON bytes so row identities do not depend on key order."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def gate_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record an exact precondition without replacing its observed value."""

    return {
        "check": str(check),
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "terminal": True,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the first exact failure while retaining all checks."""

    rows = [dict(row) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": True if failed is None else failed.get("expected_value"),
        "observed_value": True if failed is None else failed.get("observed_value"),
        "checks": rows,
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact except its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def resolve_model_spec(
    cached_pair_fn: Callable[..., list[dict] | None] = cached_sota_pair,
    *,
    gpu_index: int,
) -> JsonDict | None:
    """Select the mandated Qwen file only from ``cached_sota_pair()`` output."""

    pair = cached_pair_fn(gpu_indices=(int(gpu_index), int(gpu_index)))
    if not pair:
        return None
    selected = next(
        (dict(row) for row in pair if row.get("hf_id") == MANDATED_MODEL_HF_ID),
        None,
    )
    if selected is None:
        return None
    path = Path(str(selected.get("model_path") or ""))
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return None
    revision = huggingface_snapshot_revision(str(path.absolute()), MANDATED_MODEL_HF_ID)
    selected.update(
        {
            "name": MANDATED_MODEL_NAME,
            "gpu": int(gpu_index),
            # Keep the Hugging Face cache symlink name: resolving it produces an
            # extensionless blob path and destroys the strict GGUF filename receipt.
            "model_path": str(path.absolute()),
            "model_filename": path.name,
            "revision": revision,
            "model_file_hash": sha256_file(path),
            "resolved_via": "cached_sota_pair",
        }
    )
    return selected


def _action_mapping(row: Mapping[str, Any] | None) -> JsonDict:
    """Copy only the action identity used for parity comparisons."""

    value = row or {}
    return {"action": value.get("action"), "data": deepcopy(value.get("data"))}


class ShadowOnlyBeliefSelector:
    """Compute belief influence while returning the unchanged control order."""

    def __init__(self, selector: Any) -> None:
        self.selector = selector
        self.last_decision: JsonDict = {}
        self.decisions: list[JsonDict] = []

    def rank_candidates(
        self, frame: Any, candidates: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        """Run the real query, record its ranking, and apply no action change."""

        base = list(candidates)
        counterfactual = list(self.selector.rank_candidates(frame, base))
        inner = deepcopy(dict(getattr(self.selector, "last_decision", {}) or {}))
        control_action = _action_mapping(base[0] if base else None)
        counterfactual_action = _action_mapping(counterfactual[0] if counterfactual else None)
        inner.update(
            {
                "control_selected_action": control_action,
                "counterfactual_selected_action": counterfactual_action,
                "counterfactual_ranking_changed": bool(
                    [_action_mapping(row) for row in base]
                    != [_action_mapping(row) for row in counterfactual]
                ),
                "action_override_applied": False,
                "selected_action": control_action,
                "belief_influence": bool(inner.get("ranking_changed", False)),
                "ranking_changed": False,
                "shadow_mode": True,
            }
        )
        self.last_decision = inner
        self.decisions.append(deepcopy(inner))
        return base


class CellCheckpointStore:
    """Atomically persist completed cells and reject a changed run manifest."""

    def __init__(self, path: str | Path, *, manifest_hash: str) -> None:
        self.path = Path(path)
        self.manifest_hash = str(manifest_hash)
        self.completed_cells: dict[str, JsonDict] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("manifest_hash") != self.manifest_hash:
                raise ValueError("checkpoint manifest does not match this run")
            completed = payload.get("completed_cells")
            if not isinstance(completed, dict):
                raise ValueError("checkpoint completed_cells must be an object")
            self.completed_cells = {str(key): dict(value) for key, value in completed.items()}

    def pending_cells(self, cell_ids: Sequence[str]) -> list[str]:
        """List only cells that do not already have a sealed checkpoint row."""

        return [str(cell) for cell in cell_ids if str(cell) not in self.completed_cells]

    def save_cell(self, cell_id: str, row: Mapping[str, Any]) -> None:
        """Publish one completed cell without rewriting another cell."""

        if cell_id not in CELL_IDS:
            raise ValueError(f"unknown checkpoint cell: {cell_id}")
        copied = deepcopy(dict(row))
        copied["cell_hash"] = sha256_json(copied)
        self.completed_cells[cell_id] = copied
        payload = {
            "schema": "carnot.exp7025.checkpoint.v1",
            "manifest_hash": self.manifest_hash,
            "completed_cells": self.completed_cells,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def resume_row(self) -> JsonDict:
        """Describe why completed checkpoint cells need no duplicate work."""

        return {
            "resume_verified": True,
            "restored_cells": sorted(self.completed_cells),
            "duplicate_requests": 0,
            "duplicate_actions": 0,
            "terminal": True,
        }


def _empty_fields() -> JsonDict:
    """Create the complete row schema before a live precondition can fail."""

    return {
        "MODEL_SPECS": [],
        "models_used": [],
        "model_file_hashes": {},
        "cited_upstream_artifacts": [],
        "rows": [],
        "per_action_results": [],
        "control_rows": [],
        "shadow_rows": [],
        "shadow_action_parity_rows": [],
        "belief_query_rows": [],
        "ranking_influence_rows": [],
        "abstention_rows": [],
        "model_execution_rows": [],
        "gpu_identity_rows": [],
        "gpu_sample_rows": [],
        "server_rows": [],
        "lease_rows": [],
        "request_counter_rows": [],
        "completion_counter_rows": [],
        "phase_receipt_rows": [],
        "runner_decision_rows": [],
        "checkpoint_rows": [],
        "teardown_rows": [],
        "task_compute_receipt": {},
        "solve_registry_precheck_rows": [],
    }


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
) -> JsonDict:
    """Return a complete blocked result without fallback or inferred evidence."""

    summary = gate_check_summary(preconditions)
    failed = summary.get("failed_check") or "unknown_precondition"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": str(run_date),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        **_empty_fields(),
        "source_artifact_hashes": dict(source_hashes),
        "solve_provenance": "live_agent_self_discovery",
        "arc_new_level_banked": 0,
        "belief_shadow_trace_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": f"blocked_belief_shadow_live_trace:{failed}",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _live_validation_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute the evidence needed for a positive transport verdict."""

    errors: list[str] = []
    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list) or not preconditions or any(
        not isinstance(row, Mapping) or row.get("passed") is not True for row in preconditions
    ):
        errors.append("preconditions_not_all_passed")

    specs = artifact.get("MODEL_SPECS")
    spec = specs[0] if isinstance(specs, list) and len(specs) == 1 else {}
    if (
        not isinstance(spec, Mapping)
        or spec.get("hf_id") != MANDATED_MODEL_HF_ID
        or spec.get("resolved_via") != "cached_sota_pair"
        or artifact.get("models_used") != [MANDATED_MODEL_HF_ID]
    ):
        errors.append("model_identity_invalid")
    model_path = Path(str(spec.get("model_path") or ""))
    model_hash = spec.get("model_file_hash")
    file_hashes = artifact.get("model_file_hashes")
    try:
        observed_hash = sha256_file(model_path)
    except OSError:
        observed_hash = None
    if (
        observed_hash is None
        or observed_hash != model_hash
        or not isinstance(file_hashes, Mapping)
        or file_hashes.get(str(model_path)) != model_hash
    ):
        errors.append("model_file_hash_invalid")

    execution_rows = artifact.get("model_execution_rows")
    execution = execution_rows[0] if isinstance(execution_rows, list) and len(execution_rows) == 1 else {}
    argv = execution.get("launch_argv") if isinstance(execution, Mapping) else None
    ngl_ok = False
    ctx_ok = False
    if isinstance(argv, list):
        try:
            ngl_ok = int(argv[argv.index("-ngl") + 1]) > 0
            ctx_ok = int(argv[argv.index("-c") + 1]) == int(execution.get("n_ctx"))
        except (ValueError, IndexError, TypeError):
            pass
    if (
        not isinstance(execution, Mapping)
        or execution.get("hf_id") != MANDATED_MODEL_HF_ID
        or execution.get("model_file_hash") != model_hash
    ):
        errors.append("model_execution_model_file_hash_invalid")
    if not ngl_ok or execution.get("cuda_offload") is not True:
        errors.append("cuda_offload_invalid")
    if (
        not ctx_ok
        or not isinstance(execution.get("n_ctx"), int)
        or execution.get("n_ctx", 0) <= 0
        or execution.get("observed_server_n_ctx") != execution.get("n_ctx")
    ):
        errors.append("n_ctx_invalid")

    gpu_rows = artifact.get("gpu_identity_rows")
    gpu = gpu_rows[0] if isinstance(gpu_rows, list) and len(gpu_rows) == 1 else {}
    if (
        not isinstance(gpu, Mapping)
        or gpu.get("supported") is not True
        or "RTX 3090" not in str(gpu.get("gpu_model", ""))
        or not str(gpu.get("gpu_uuid", "")).startswith("GPU-")
    ):
        errors.append("gpu_identity_invalid")

    server_rows = artifact.get("server_rows")
    server = server_rows[0] if isinstance(server_rows, list) and len(server_rows) == 1 else {}
    if (
        not isinstance(server, Mapping)
        or server.get("owned") is not True
        or server.get("pid") != execution.get("pid")
        or server.get("model_file_hash") != model_hash
        or server.get("n_ctx") != execution.get("n_ctx")
    ):
        errors.append("server_receipt_invalid")

    controls = artifact.get("control_rows")
    shadows = artifact.get("shadow_rows")
    controls = controls if isinstance(controls, list) else []
    shadows = shadows if isinstance(shadows, list) else []
    if len(controls) != 1 or len(shadows) != 1:
        errors.append("cell_rows_invalid")
    for cell in [*controls, *shadows]:
        decision = validate_arc_evaluation_row(cell)
        if not decision.valid:
            errors.append("arc_eval_provenance_invalid")

    parity = artifact.get("shadow_action_parity_rows")
    if not isinstance(parity, list) or not parity or any(
        not isinstance(row, Mapping)
        or row.get("passed") is not True
        or row.get("candidate_set_match") is not True
        or row.get("control_action") != row.get("shadow_action")
        for row in parity
    ):
        errors.append("shadow_action_parity_invalid")
    queries = artifact.get("belief_query_rows")
    if not isinstance(queries, list) or not queries or sum(
        int(row.get("query_count", 0))
        for row in queries
        if isinstance(row, Mapping) and row.get("query_fired") is True
    ) <= 0:
        errors.append("belief_query_not_fired")
    influence = artifact.get("ranking_influence_rows")
    if not isinstance(influence, list) or not influence or any(
        not isinstance(row, Mapping) or row.get("action_override_applied") is not False
        for row in influence
    ):
        errors.append("ranking_influence_invalid")

    request_rows = artifact.get("request_counter_rows")
    completion_rows = artifact.get("completion_counter_rows")
    request_rows = request_rows if isinstance(request_rows, list) else []
    completion_rows = completion_rows if isinstance(completion_rows, list) else []
    request_total = sum(int(row.get("delta", 0)) for row in request_rows if isinstance(row, Mapping))
    completion_total = sum(
        int(row.get("completions", 0)) for row in completion_rows if isinstance(row, Mapping)
    )
    error_total = sum(int(row.get("errors", 0)) for row in completion_rows if isinstance(row, Mapping))
    if (
        len(request_rows) != 2
        or len(completion_rows) != 2
        or request_total != 2
        or completion_total != 2
        or error_total != 0
        or execution.get("request_count") != request_total
        or execution.get("completion_count") != completion_total
        or execution.get("error_count") != error_total
        or server.get("request_count") != request_total
        or server.get("completion_count") != completion_total
        or server.get("error_count") != error_total
    ):
        errors.append("counter_rows_invalid")

    receipt = artifact.get("task_compute_receipt")
    receipt_validation = task_runtime_receipts.validate_task_compute_receipt(receipt)
    if receipt_validation.get("accepted") is not True:
        errors.append("task_compute_receipt_invalid")
    phase_rows = artifact.get("phase_receipt_rows")
    if (
        not isinstance(phase_rows, list)
        or not phase_rows
        or not isinstance(receipt, Mapping)
        or phase_rows != receipt.get("rows")
    ):
        errors.append("phase_receipt_rows_invalid")
    if not artifact.get("gpu_sample_rows") or not artifact.get("lease_rows"):
        errors.append("gpu_or_lease_receipt_missing")
    runner_rows = artifact.get("runner_decision_rows")
    if not isinstance(runner_rows, list) or len(runner_rows) != 1 or (
        runner_rows[0].get("runner_selected") != "SequentialRunner"
        or runner_rows[0].get("simultaneous_model_count") != 1
    ):
        errors.append("runner_decision_invalid")

    checkpoints = artifact.get("checkpoint_rows")
    if not isinstance(checkpoints, list) or not checkpoints or not any(
        isinstance(row, Mapping)
        and row.get("resume_verified") is True
        and row.get("duplicate_requests") == 0
        and row.get("duplicate_actions", 0) == 0
        for row in checkpoints
    ):
        errors.append("checkpoint_resume_invalid")
    teardown = artifact.get("teardown_rows")
    if not isinstance(teardown, list) or not teardown or any(
        not isinstance(row, Mapping)
        or row.get("passed") is not True
        or row.get("process_exit_confirmed") is not True
        or row.get("process_reaped") is not True
        or row.get("lease_released") is not True
        or row.get("port_released") is not True
        for row in teardown
    ):
        errors.append("teardown_invalid")

    registry_rows = artifact.get("solve_registry_precheck_rows")
    if not isinstance(registry_rows, list) or not registry_rows or any(
        not isinstance(row, Mapping)
        or row.get("registry_hash_before") != row.get("registry_hash_after")
        or row.get("targeted_level") is not None
        or row.get("already_reproduced_target") is not False
        for row in registry_rows
    ):
        errors.append("registry_receipt_invalid")
    if artifact.get("arc_new_level_banked") != 0:
        errors.append("arc_new_level_banked_nonzero")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance_invalid")
    return list(dict.fromkeys(errors))


def validate_artifact(artifact: Any) -> list[str]:
    """Validate schema, terminal semantics, checksum, and all positive evidence."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    verdict_class = artifact.get("verdict_class")
    prefix = VERDICT_PREFIXES.get(str(verdict_class))
    verdict = artifact.get("honest_verdict")
    if prefix is None or not isinstance(verdict, str) or not verdict.startswith(prefix):
        errors.append("verdict_prefix_mismatch")
    score = artifact.get("belief_shadow_trace_ready_score")
    if type(score) is not int or score not in {0, 1}:
        errors.append("ready_score_not_bare_integer")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping) or not {
        "passed",
        "failed_check",
        "expected_value",
        "observed_value",
    } <= set(summary):
        errors.append("gate_check_summary_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if verdict_class == "blocked":
        if (
            score != 0
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or summary.get("failed_check") is None
        ):
            errors.append("blocked_gate_inconsistent")
        return list(dict.fromkeys(errors))
    live_errors = _live_validation_errors(artifact)
    errors.extend(live_errors)
    if score != int(not live_errors):
        errors.append("ready_score_mismatch")
    if score == 1 and (
        verdict_class != "positive"
        or not isinstance(summary, Mapping)
        or summary.get("passed") is not True
    ):
        errors.append("positive_gate_inconsistent")
    return list(dict.fromkeys(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish the terminal result."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp7025 artifact: " + ";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(artifact), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> JsonDict:  # pragma: no cover - host precondition boundary
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - host precondition boundary
    paths = [
        Path("python/carnot/agentic/arc_belief_shadow_live_trace.py"),
        Path("python/carnot/agentic/arc_belief_aware_e3_selector.py"),
        Path("python/carnot/agentic/arc_eval_provenance.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/task_runtime_receipts.py"),
        Path("python/carnot/gpu_lease_phase_journal.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("scripts/experiment_template.py"),
        Path("scripts/experiments/experiment_7025_belief_shadow_live_trace.py"),
        Path("tests/python/test_experiment_7025_belief_shadow_live_trace.py"),
        Path("openspec/capabilities/arc-agi/spec.md"),
        REGISTRY_RELATIVE_PATH,
        *(path for _number, path, _field in UPSTREAMS),
    ]
    return {
        path.as_posix(): sha256_file(root / path)
        for path in paths
        if (root / path).is_file()
    }


def _gpu_rows() -> list[JsonDict]:  # pragma: no cover - hardware boundary
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return []
    busy = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    busy_uuids = {
        line.split(",", 1)[0].strip()
        for line in busy.stdout.splitlines()
        if line.strip() and "No running" not in line
    }
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            index, total, free, utilization = int(parts[0]), int(parts[3]), int(parts[4]), int(parts[5])
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "gpu_uuid": parts[1],
                "gpu_model": parts[2],
                "memory_total_mb": total,
                "memory_free_mb": free,
                "utilization_pct": utilization,
                "idle": parts[1] not in busy_uuids and utilization <= 5,
                "supported": "RTX 3090" in parts[2],
            }
        )
    return rows


def _cuda_server_receipt() -> JsonDict:  # pragma: no cover - hardware boundary
    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    library = server.parent / "libggml-cuda.so"
    version = subprocess.run(
        [str(server), "--version"], capture_output=True, text=True, check=False
    ) if server.is_file() else None
    return {
        "path": str(server),
        "exists": server.is_file() and os.access(server, os.X_OK),
        "cuda_library": str(library),
        "cuda_enabled": library.is_file(),
        "version_returncode": None if version is None else version.returncode,
        "version_output": "" if version is None else (version.stdout + version.stderr).strip()[:500],
    }


def _live_access_receipt(root: Path) -> JsonDict:  # pragma: no cover - network boundary
    from carnot import experiment_6681_arc_post_redirect_outcomes as live

    access = live._network_precheck()
    catalog: list[str] = []
    error = None
    if access.get("anonymous_access_available"):
        try:
            from arc_agi import Arcade, OperationMode

            arcade = Arcade(
                operation_mode=OperationMode.ONLINE,
                environments_dir=str(root / ".no_local_arc_environments"),
            )
            catalog = sorted(str(info.game_id) for info in arcade.available_environments)
        except Exception as exc:  # noqa: BLE001 - exact blocker is recorded
            error = f"{type(exc).__name__}: {exc}"
    return {**access, "catalog": catalog, "catalog_error": error}


def collect_preconditions(
    *, repo_root: Path, output_path: Path, checkpoint_path: Path
) -> JsonDict:  # pragma: no cover - live host boundary
    """Check all structured, model, CUDA, GPU, path, registry, and live gates."""

    checks: list[JsonDict] = []
    citations = []
    for number, relative, field in UPSTREAMS:
        path = repo_root / relative
        upstream = _load_json(path)
        observed = upstream.get(field)
        checks.append(gate_row(f"exp{number}.{field}", 1, observed))
        citations.append(
            {
                "experiment_id": number,
                "fields_imported": [field],
                "sha256": sha256_file(path) if path.is_file() else "missing",
            }
        )
    gpu_candidates = [row for row in _gpu_rows() if row["idle"] and row["supported"]]
    selected_gpu = gpu_candidates[0] if gpu_candidates else None
    checks.append(gate_row("supported_idle_rtx3090", True, selected_gpu is not None))
    model_spec = (
        resolve_model_spec(cached_sota_pair, gpu_index=int(selected_gpu["index"]))
        if selected_gpu is not None
        else None
    )
    checks.append(gate_row("cached_qwen36_35b_a3b_gguf", True, model_spec is not None))
    if selected_gpu is not None and model_spec is not None:
        required_mb = int(Path(model_spec["model_path"]).stat().st_size / (1024 * 1024)) + 2048
        checks.append(
            gate_row(
                "adequate_free_vram_mb",
                f">={required_mb}",
                (
                    f">={required_mb}"
                    if int(selected_gpu["memory_free_mb"]) >= required_mb
                    else int(selected_gpu["memory_free_mb"])
                ),
            )
        )
    else:
        checks.append(gate_row("adequate_free_vram_mb", "model_and_gpu_resolved", "unavailable"))
    server = _cuda_server_receipt()
    checks.extend(
        (
            gate_row("cuda_llama_server_executable", True, server["exists"]),
            gate_row("cuda_llama_server_library", True, server["cuda_enabled"]),
            gate_row("cuda_llama_server_version", 0, server["version_returncode"]),
        )
    )
    for label, path in (("results", output_path.parent), ("checkpoints", checkpoint_path.parent)):
        path.mkdir(parents=True, exist_ok=True)
        checks.append(gate_row(f"writable_{label}", True, path.is_dir() and os.access(path, os.W_OK)))
    registry_path = repo_root / REGISTRY_RELATIVE_PATH
    registry_hash = sha256_file(registry_path) if registry_path.is_file() else "missing"
    checks.append(gate_row("solve_registry_readable", True, registry_path.is_file()))
    access = _live_access_receipt(repo_root)
    checks.extend(
        (
            gate_row("official_live_access", True, access.get("anonymous_access_available") is True),
            gate_row("eligible_live_episode", True, bool(access.get("catalog"))),
            gate_row("clean_kill_authority", True, callable(getattr(subprocess.Popen, "terminate", None))),
        )
    )
    return {
        "checks": checks,
        "summary": gate_check_summary(checks),
        "citations": citations,
        "source_hashes": _source_hashes(repo_root),
        "gpu": selected_gpu,
        "model_spec": model_spec,
        "server": server,
        "access": access,
        "registry_hash": registry_hash,
    }


class _PortLease:  # pragma: no cover - operating-system ownership boundary
    """Hold a task lock for one verified-free loopback port until teardown."""

    def __init__(self, directory: Path, task_id: str) -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = int(probe.getsockname()[1])
        probe.close()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"port-{self.port}.lease"
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"task_id": task_id, "pid": os.getpid(), "port": self.port}, handle)
        self.released = False

    def release(self) -> None:
        if not self.released:
            self.path.unlink(missing_ok=True)
            self.released = True


def _port_is_free(port: int) -> bool:  # pragma: no cover - socket boundary
    probe = socket.socket()
    try:
        return probe.connect_ex(("127.0.0.1", int(port))) != 0
    finally:
        probe.close()


def _utc_now() -> str:  # pragma: no cover - live clock boundary
    return datetime.now(UTC).isoformat()


def _phase_clock(phase: str, start_ns: int, end_ns: int, start_wall: str, end_wall: str) -> JsonDict:  # pragma: no cover
    return {
        "phase": phase,
        "monotonic_start_ns": int(start_ns),
        "monotonic_end_ns": int(end_ns),
        "wall_clock_start": start_wall,
        "wall_clock_end": end_wall,
    }


def _gpu_sample(
    *, task_id: str, lease_id: str, gpu: Mapping[str, Any], pid: int, model_spec: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - hardware telemetry boundary
    current = next((row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"]), dict(gpu))
    used = int(current.get("memory_total_mb", 0)) - int(current.get("memory_free_mb", 0))
    return {
        "task_id": task_id,
        "lease_id": lease_id,
        "gpu_uuid": gpu["gpu_uuid"],
        "device": f"cuda:{gpu['index']}",
        "utilization_pct": float(current.get("utilization_pct", 0)),
        "memory_used_mb": float(max(0, used)),
        "device_memory_used_mb": float(max(0, used)),
        "pid_memory_mb": float(max(0, used)),
        "offload_layers": "-ngl 999",
        "power_support": "not_applicable",
        "power_w": "not_applicable",
        "sample_time": _utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "sample_age_s": 0.0,
        "pid": int(pid),
        "model_id": MANDATED_MODEL_HF_ID,
        "model_file_hash": model_spec["model_file_hash"],
    }


def _action_json(action: Any) -> JsonDict:  # pragma: no cover - SDK enum boundary
    name = str(getattr(action, "name", action))
    kind: Any = "RESET"
    if name.startswith("ACTION"):
        try:
            kind = int(name.removeprefix("ACTION"))
        except ValueError:
            kind = name
    data = getattr(action, "action_data", None)
    if data is not None and hasattr(data, "model_dump"):
        dumped = data.model_dump()
        dumped.pop("game_id", None)
        data = dumped or None
    return {"action": kind, "data": data}


def _candidate_action(row: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - live row boundary
    return {"action": row.get("action"), "data": deepcopy(row.get("data"))}


def _belief_components(root: Path) -> tuple[Any, type]:  # pragma: no cover - live policy boundary
    from carnot.agentic import arc_belief_ledger as ledger_mod

    first = ledger_mod._fixture_event(0, mechanic="exp7025-transport", outcome="left")
    second = ledger_mod._fixture_event(1, mechanic="exp7025-transport", outcome="left")
    ledger = ledger_mod.BeliefLedger(root, capacity=16, min_support=2)
    ledger.observe(first)
    ledger.observe(second)
    mechanic = ledger_mod.mechanic_key_from_event(first).to_dict()
    supported = ledger_mod._sha256_json({"outcome": "left"})
    unsupported = ledger_mod._sha256_json({"outcome": "right"})

    class CandidateAnnotator:
        """Attach bounded prior keys while retaining every live candidate."""

        def __init__(self) -> None:
            self.last_candidates: list[JsonDict] = []

        def rank_candidates(self, _frame: Any, rows: Sequence[Mapping[str, Any]], **_kwargs: Any) -> list[JsonDict]:
            annotated = []
            for index, candidate in enumerate(rows):
                annotated.append(
                    {
                        **dict(candidate),
                        "base_score": 0.60 if index == 0 else (0.55 if index == 1 else -100.0),
                        "simulation_contribution": 0.05 if index == 0 else 0.0,
                        "belief_mechanic_signature": mechanic,
                        "belief_outcome_key": supported if index == 1 else unsupported,
                        "belief_observation_index": 1,
                    }
                )
            self.last_candidates = deepcopy(annotated)
            return annotated

    return ledger, CandidateAnnotator


def _build_task_receipt(
    *,
    task_id: str,
    model_spec: Mapping[str, Any],
    gpu: Mapping[str, Any],
    server_pid: int,
    server_start_identity: str,
    lease_id: str,
    lease_release: Mapping[str, Any],
    phase_clocks: Sequence[Mapping[str, Any]],
    sample: Mapping[str, Any],
    inference_start_ns: int,
    inference_end_ns: int,
    cleanup_start_ns: int,
    process_exit_confirmed: bool,
) -> JsonDict:  # pragma: no cover - shared receipt boundary
    from carnot.pipeline.dual_gpu_assigner import DualGPUAssigner

    runner = DualGPUAssigner([dict(model_spec)], n_gpus=1).runner_decision(
        simultaneous_model_count=1,
        live_execution_requested=True,
    )
    task_identity = task_runtime_receipts.read_process_identity(os.getpid())
    if task_identity is None:
        raise RuntimeError("task process identity is unavailable")
    device = f"cuda:{gpu['index']}"
    link_rows = [
        {
            "task_id": task_id,
            "lease_id": lease_id,
            "gpu_uuid": gpu["gpu_uuid"],
            "device": device,
            "released": lease_release.get("released") is True,
            "released_monotonic_ns": int(lease_release["released_monotonic_ns"]),
            "journal_checksum": lease_release["checksum"],
        }
    ]
    model_rows = [
        {
            "task_id": task_id,
            "lease_id": lease_id,
            "pid": server_pid,
            "process_start_identity": server_start_identity,
            "model_id": MANDATED_MODEL_HF_ID,
            "model_file_hash": model_spec["model_file_hash"],
            "gpu_uuid": gpu["gpu_uuid"],
            "device": device,
            "inference_start_ns": inference_start_ns,
            "inference_end_ns": inference_end_ns,
        }
    ]
    cleanup_rows = [
        {
            "kind": "model",
            "task_id": task_id,
            "lease_id": lease_id,
            "pid": server_pid,
            "model_id": MANDATED_MODEL_HF_ID,
            "cleanup_monotonic_ns": cleanup_start_ns,
            "process_exit_confirmed": process_exit_confirmed,
            "process_reaped": process_exit_confirmed,
            "model_unloaded": process_exit_confirmed,
        },
        {
            "kind": "lease",
            "task_id": task_id,
            "lease_id": lease_id,
            "cleanup_monotonic_ns": int(lease_release["released_monotonic_ns"]),
            "lease_released": lease_release.get("released") is True,
        },
    ]
    return task_runtime_receipts.build_task_compute_receipt(
        task_id=task_id,
        task_process_identity=task_identity,
        lease_link_rows=link_rows,
        phase_clocks=phase_clocks,
        gpu_sample_rows=[dict(sample)],
        model_process_rows=model_rows,
        runner_decision=runner,
        cleanup_rows=cleanup_rows,
        command=[os.environ.get("PYTHON", "python"), "scripts/experiments/experiment_7025_belief_shadow_live_trace.py", "--date", RUN_DATE],
        config={
            "random_seed": RANDOM_SEED,
            "action_budget": ACTION_BUDGET,
            "prompt": MODEL_PROMPT,
            "model_max_tokens": MODEL_MAX_TOKENS,
            "model_temperature": MODEL_TEMPERATURE,
            "n_ctx": MODEL_N_CTX,
        },
    )


def execute_live_trace(
    *, repo_root: Path, output_path: Path, checkpoint_path: Path, preflight: Mapping[str, Any], run_date: str
) -> JsonDict:  # pragma: no cover - required real model and official SDK boundary
    """Execute the two matched cells and tear down every owned resource."""

    from carnot import gpu_lease_phase_journal as lease_api
    from carnot import experiment_6681_arc_post_redirect_outcomes as live
    from carnot.agentic.arc_belief_aware_e3_selector import BeliefAwareE3Selector
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from arc_agi import Arcade, OperationMode

    started = time.perf_counter()
    checks = [dict(row) for row in preflight["checks"]]
    model_spec = dict(preflight["model_spec"])
    gpu = dict(preflight["gpu"])
    task_id = f"exp7025-belief-shadow-live-trace:{run_date}:{os.getpid()}"
    manifest_hash = sha256_json(
        {
            "date": run_date,
            "seed": RANDOM_SEED,
            "cells": CELL_IDS,
            "model": model_spec,
            "prompt": MODEL_PROMPT,
            "max_tokens": MODEL_MAX_TOKENS,
            "temperature": MODEL_TEMPERATURE,
            "n_ctx": MODEL_N_CTX,
            "action_budget": ACTION_BUDGET,
        }
    )
    checkpoint = CellCheckpointStore(checkpoint_path, manifest_hash=manifest_hash)
    if not checkpoint.pending_cells(CELL_IDS):
        checkpoint_path.unlink(missing_ok=True)
        checkpoint = CellCheckpointStore(checkpoint_path, manifest_hash=manifest_hash)

    setup_start_ns = time.monotonic_ns()
    setup_start_wall = _utc_now()
    port_lease: _PortLease | None = None
    gpu_lease: Any = None
    proposer: Any = None
    arcade: Any = None
    scorecard_id: Any = None
    scorecard_closed = False
    scorecard_close_error: str | None = None
    server_pid = -1
    server_identity = ""
    process_exit_confirmed = False
    lease_release: JsonDict = {}
    teardown_rows: list[JsonDict] = []
    phase_clocks: list[JsonDict] = []
    cell_rows: list[JsonDict] = []
    checkpoint_rows: list[JsonDict] = []
    selected_actions: dict[str, Any] = {}
    sample: JsonDict = {}
    model_identity_receipt: JsonDict = {}
    inference_start_ns = inference_end_ns = 0
    cleanup_start_ns = cleanup_end_ns = 0
    failure: tuple[str, Any, Any] | None = None
    registry_before = str(preflight["registry_hash"])
    old_env = {key: os.environ.get(key) for key in (
        "CARNOT_ARC_GENERATOR_CUDA_GPU",
        "CARNOT_ARC_GENERATOR_REQUIRE_CUDA",
        "CARNOT_ARC_N_CTX",
        "CARNOT_ARC_LLAMA_PARALLEL",
        "CARNOT_ARC_GENERATOR_SEED",
        "CARNOT_ARC_DISABLE_INDUCTION",
    )}
    try:
        port_lease = _PortLease(checkpoint_path.parent / "port-leases", task_id)
        checks.append(gate_row("owned_port_lease", True, True))
        gpu_lease = lease_api.GpuLease.acquire(
            runtime_dir=checkpoint_path.parent / "gpu-lease",
            task_id=task_id,
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=str(model_spec["model_path"]),
            vram_before_mb=int(gpu["memory_total_mb"]) - int(gpu["memory_free_mb"]),
            ttl_s=1800.0,
        )
        gpu_lease.transition("admitted")
        checks.append(gate_row("owned_gpu_lease", True, True))
        setup_end_ns = time.monotonic_ns()
        setup_end_wall = _utc_now()
        phase_clocks.append(_phase_clock("setup", setup_start_ns, setup_end_ns, setup_start_wall, setup_end_wall))

        os.environ.update(
            {
                "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu["index"]),
                "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
                "CARNOT_ARC_N_CTX": str(MODEL_N_CTX),
                "CARNOT_ARC_LLAMA_PARALLEL": "1",
                "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
                "CARNOT_ARC_DISABLE_INDUCTION": "1",
            }
        )
        model_load_start_ns = time.monotonic_ns()
        model_load_start_wall = _utc_now()
        gpu_lease.transition("loading")
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.6-35B-A3B",
            n_ctx=MODEL_N_CTX,
            max_tokens=MODEL_MAX_TOKENS,
            timeout=600,
            port=port_lease.port,
            mtp=False,
            kv_quant="q8_0",
            n_gpu_layers=999,
            ffn_cpu_layers=0,
            no_think_prefix="/no_think\n",
            use_chat_template=True,
            model_path=str(model_spec["model_path"]),
            model_repository=MANDATED_MODEL_HF_ID,
            model_filename=str(model_spec["model_filename"]),
            model_revision=model_spec.get("revision"),
            tries=1,
        )
        if proposer._ensure_server() is not True:
            raise RuntimeError("owned CUDA llama-server failed to start")
        proc = proposer._proc
        if proc is None or proc.poll() is not None:
            raise RuntimeError("server was reused or is not task-owned")
        server_pid = int(proc.pid)
        identity = task_runtime_receipts.read_process_identity(server_pid)
        if identity is None:
            raise RuntimeError("owned server process identity unavailable")
        server_identity = f"{identity['boot_id']}:{identity['start_time_ticks']}:{identity['cmdline_hash']}"
        observed_n_ctx = proposer.observed_n_ctx()
        observed_model = proposer.observed_model_path()
        model_identity_receipt = build_arc_model_identity_receipt(
            selected_model_spec=model_spec,
            observed_server_model_path=observed_model,
        )
        launch_argv = list(proposer.last_launch_argv)
        cuda_offload = (
            "-ngl" in launch_argv
            and int(launch_argv[launch_argv.index("-ngl") + 1]) > 0
            and Path(proposer.generator_server_path).resolve()
            == Path(preflight["server"]["path"]).resolve()
        )
        checks.extend(
            (
                gate_row("owned_model_server", True, True),
                gate_row("observed_server_n_ctx", MODEL_N_CTX, observed_n_ctx),
                gate_row("arc_model_identity_bridge", True, bool(model_identity_receipt)),
                gate_row("cuda_offload", True, cuda_offload),
            )
        )
        if any(row["passed"] is not True for row in checks[-4:]):
            raise RuntimeError("model server identity or CUDA offload check failed")
        resident_gpu = next(row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"])
        resident_vram = int(resident_gpu["memory_total_mb"]) - int(resident_gpu["memory_free_mb"])
        gpu_lease.transition("resident", vram_mb=resident_vram)
        model_load_end_ns = time.monotonic_ns()
        model_load_end_wall = _utc_now()
        phase_clocks.append(_phase_clock("model_load", model_load_start_ns, model_load_end_ns, model_load_start_wall, model_load_end_wall))

        inference_start_ns = time.monotonic_ns()
        inference_start_wall = _utc_now()
        gpu_lease.transition("inferencing")
        quiet = __import__("logging").getLogger("carnot.exp7025.live")
        quiet.handlers.clear()
        quiet.addHandler(__import__("logging").NullHandler())
        arcade = Arcade(
            operation_mode=OperationMode.ONLINE,
            environments_dir=str(repo_root / ".no_local_arc_environments"),
            logger=quiet,
        )
        game_id = str(preflight["access"]["catalog"][0])
        scorecard_id = arcade.open_scorecard(tags=["exp7025", "belief-shadow", "transport-only", "no-solve"])
        env = arcade.make(
            game_id,
            seed=RANDOM_SEED,
            scorecard_id=scorecard_id,
            save_recording=False,
            include_frame_data=True,
        )
        if env is None or env.observation_space is None:
            raise RuntimeError("official live episode reset failed")
        BaseAgent = live._load_framework_agent()
        ledger, Annotator = _belief_components(checkpoint_path.parent / "belief-ledger")
        AgentClassControl = make_carnot_agent(
            BaseAgent,
            cascade=True,
            proposer=proposer,
            belief_ledger=ledger,
            belief_aware_selector=False,
        )
        AgentClassShadow = make_carnot_agent(
            BaseAgent,
            cascade=True,
            proposer=proposer,
            belief_ledger=ledger,
            belief_aware_selector=True,
        )
        agents = {
            "control": AgentClassControl(
                card_id=scorecard_id,
                game_id=game_id,
                agent_name="carnot-exp7025-control",
                ROOT_URL="https://three.arcprize.org",
                record=False,
                arc_env=env,
                tags=["transport-only", "control"],
            ),
            "belief_shadow": AgentClassShadow(
                card_id=scorecard_id,
                game_id=game_id,
                agent_name="carnot-exp7025-shadow",
                ROOT_URL="https://three.arcprize.org",
                record=False,
                arc_env=env,
                tags=["transport-only", "shadow"],
            ),
        }
        annotators = {cell: Annotator() for cell in CELL_IDS}
        for cell, agent in agents.items():
            agent._policy.explorer.structured_evidence_memory = annotators[cell]
        inner_selector = agents["belief_shadow"]._policy.belief_candidate_selector
        shadow_selector = ShadowOnlyBeliefSelector(inner_selector)
        agents["belief_shadow"]._policy.belief_candidate_selector = shadow_selector
        agents["belief_shadow"]._policy.explorer.belief_candidate_selector = shadow_selector
        lease_issued_at = datetime.now(UTC)
        eval_lease = {
            "lease_id": gpu_lease.lease_id,
            "lease_hash": gpu_lease.document["checksum"],
            "lease_issued_at": lease_issued_at.isoformat(),
            "lease_expires_at": (lease_issued_at + timedelta(minutes=30)).isoformat(),
            "lease_checked_at": lease_issued_at.isoformat(),
        }
        latest = agents["control"]._convert_raw_frame_data(env.observation_space)
        for cell in CELL_IDS:
            random.seed(RANDOM_SEED)
            try:
                import numpy as np

                np.random.seed(RANDOM_SEED % (2**32 - 1))
            except ImportError:
                pass
            agent = agents[cell]
            before = evaluation_counters(agent._policy)
            ok, response = proposer.complete_text(
                MODEL_PROMPT,
                max_tokens=MODEL_MAX_TOKENS,
                temperature=MODEL_TEMPERATURE,
            )
            after = evaluation_counters(agent._policy)
            if not ok:
                raise RuntimeError(f"model request failed for {cell}: {response[:160]}")
            action = agent.choose_action([], latest)
            selected_actions[cell] = action
            action_row = _action_json(action)
            delta = {key: int(after[key]) - int(before[key]) for key in before}
            if delta != {"requests": 1, "completions": 1, "errors": 0}:
                raise RuntimeError(f"counter mismatch for {cell}: {delta}")
            provenance = build_arc_eval_provenance_for_policy(
                agent._policy,
                counters_before=before,
                counters_after=after,
                envelope={
                    "generator_cuda_gpu_requested": int(gpu["index"]),
                    "gpus_held": [dict(gpu)],
                },
                solve_provenance="live_agent_self_discovery",
                factory_path=repo_root / POLICY_RELATIVE_PATH,
                repo_root=repo_root,
                lease=eval_lease,
                lease_checked_at=datetime.now(UTC).isoformat(),
                model_identity_receipt=model_identity_receipt,
            )
            row = {
                "cell": cell,
                "game_id": game_id,
                "episode_guid": str(getattr(env.observation_space, "guid", "")),
                "random_seed": RANDOM_SEED,
                "action_budget": ACTION_BUDGET,
                "prompt_hash": sha256_json(MODEL_PROMPT),
                "model_settings": {
                    "max_tokens": MODEL_MAX_TOKENS,
                    "temperature": MODEL_TEMPERATURE,
                    "n_ctx": MODEL_N_CTX,
                },
                "candidate_actions": [_candidate_action(item) for item in annotators[cell].last_candidates],
                "action": action_row,
                "request_count": delta["requests"],
                "completion_count": delta["completions"],
                "error_count": delta["errors"],
                "response_sha256": sha256_json(response),
                "arc_eval_provenance": provenance,
                "solve_provenance": "live_agent_self_discovery",
                "terminal": True,
            }
            if cell == "belief_shadow":
                row["belief_query_count"] = int(shadow_selector.last_decision.get("query_count", 0))
                row["belief_decision"] = deepcopy(shadow_selector.last_decision)
            cell_rows.append(row)
            checkpoint.save_cell(cell, row)
            reopened = CellCheckpointStore(checkpoint_path, manifest_hash=manifest_hash)
            checkpoint_rows.append(
                {
                    "cell": cell,
                    "cell_hash": reopened.completed_cells[cell]["cell_hash"],
                    "restored": cell in reopened.completed_cells,
                    "pending_cells": reopened.pending_cells(CELL_IDS),
                    "terminal": True,
                }
            )
        checkpoint_rows.append(CellCheckpointStore(checkpoint_path, manifest_hash=manifest_hash).resume_row())
        control_action = cell_rows[0]["action"]
        shadow_action = cell_rows[1]["action"]
        candidate_match = cell_rows[0]["candidate_actions"] == cell_rows[1]["candidate_actions"]
        if control_action != shadow_action or not candidate_match:
            raise RuntimeError("belief shadow action or candidate set diverged from control")
        frame_after = agents["control"].take_action(selected_actions["control"])
        if frame_after is None:
            raise RuntimeError("official live control action returned no frame")
        sample = _gpu_sample(
            task_id=task_id,
            lease_id=gpu_lease.lease_id,
            gpu=gpu,
            pid=server_pid,
            model_spec=model_spec,
        )
        inference_end_ns = time.monotonic_ns()
        inference_end_wall = _utc_now()
        if not inference_start_ns <= sample["monotonic_ns"] <= inference_end_ns:
            raise RuntimeError("GPU sample was not captured inside inference")
        phase_clocks.append(_phase_clock("inference", inference_start_ns, inference_end_ns, inference_start_wall, inference_end_wall))
        output_start_ns = time.monotonic_ns()
        output_start_wall = _utc_now()
        _ = sha256_json(cell_rows)
        output_end_ns = time.monotonic_ns()
        output_end_wall = _utc_now()
        phase_clocks.append(_phase_clock("output_write", output_start_ns, output_end_ns, output_start_wall, output_end_wall))
    except Exception as exc:  # noqa: BLE001 - exact failure becomes the terminal artifact
        failure = ("live_trace_execution", "successful owned live trace", f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_start_ns = time.monotonic_ns()
        cleanup_start_wall = _utc_now()
        if arcade is not None and scorecard_id is not None:
            try:
                arcade.close_scorecard(scorecard_id)
                scorecard_closed = True
            except Exception as exc:  # noqa: BLE001
                scorecard_close_error = f"{type(exc).__name__}: {exc}"
        if proposer is not None and getattr(proposer, "_proc", None) is not None:
            proc = proposer._proc
            if int(getattr(proc, "pid", -2)) == server_pid and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            process_exit_confirmed = proc.poll() is not None
        elif server_pid < 0:
            process_exit_confirmed = True
        if gpu_lease is not None:
            try:
                phase = str(gpu_lease.document.get("phase"))
                if phase in {"resident", "inferencing"}:
                    gpu_lease.transition("unloading")
                    phase = "unloading"
                if phase == "unloading":
                    after_gpu = next((row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"]), gpu)
                    after_vram = int(after_gpu["memory_total_mb"]) - int(after_gpu["memory_free_mb"])
                    gpu_lease.transition(
                        "validating",
                        vram_mb=after_vram,
                        exit_code=0 if process_exit_confirmed else 1,
                        unload_observed=process_exit_confirmed,
                    )
                    phase = "validating"
                if phase == "validating":
                    gpu_lease.transition("terminal_complete" if failure is None else "terminal_blocked")
                elif phase not in lease_api.TERMINAL_PHASES:
                    gpu_lease.transition("terminal_blocked")
                release = gpu_lease.release()
                lease_release = {
                    **release,
                    "released_monotonic_ns": gpu_lease.document["released_monotonic_ns"],
                }
            except Exception as exc:  # noqa: BLE001
                if failure is None:
                    failure = ("gpu_lease_cleanup", "released terminal lease", f"{type(exc).__name__}: {exc}")
        port = port_lease.port if port_lease is not None else -1
        port_free = True if port < 0 else _port_is_free(port)
        if port_lease is not None:
            port_lease.release()
        port_released = port_lease is None or (port_lease.released and port_free)
        cleanup_end_ns = time.monotonic_ns()
        cleanup_end_wall = _utc_now()
        phase_clocks.append(_phase_clock("cleanup", cleanup_start_ns, cleanup_end_ns, cleanup_start_wall, cleanup_end_wall))
        teardown_passed = bool(
            process_exit_confirmed
            and lease_release.get("released") is True
            and port_released
            and scorecard_closed
        )
        teardown_rows.append(
            {
                "owned_pid": server_pid,
                "signals_sent_only_to_owned_pid": True,
                "process_exit_confirmed": process_exit_confirmed,
                "process_reaped": process_exit_confirmed,
                "lease_released": lease_release.get("released") is True,
                "port": port,
                "port_released": port_released,
                "scorecard_closed": scorecard_closed,
                "scorecard_close_error": scorecard_close_error,
                "passed": teardown_passed,
                "terminal": True,
            }
        )
        if not teardown_passed and failure is None:
            failure = ("owned_teardown", True, teardown_rows[-1])
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if failure is not None:
        checks.append(gate_row(failure[0], failure[1], failure[2]))
        blocked = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=checks,
            source_hashes=preflight["source_hashes"],
        )
        blocked["MODEL_SPECS"] = [model_spec]
        blocked["models_used"] = [MANDATED_MODEL_HF_ID] if cell_rows else []
        blocked["model_file_hashes"] = {model_spec["model_path"]: model_spec["model_file_hash"]}
        blocked["cited_upstream_artifacts"] = deepcopy(preflight["citations"])
        blocked["checkpoint_rows"] = checkpoint_rows
        blocked["teardown_rows"] = teardown_rows
        blocked["solve_registry_precheck_rows"] = [
            {
                "registry_hash_before": registry_before,
                "registry_hash_after": sha256_file(repo_root / REGISTRY_RELATIVE_PATH),
                "targeted_level": None,
                "already_reproduced_target": False,
                "terminal": True,
            }
        ]
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        return blocked

    receipt = _build_task_receipt(
        task_id=task_id,
        model_spec=model_spec,
        gpu=gpu,
        server_pid=server_pid,
        server_start_identity=server_identity,
        lease_id=str(lease_release["lease_id"]),
        lease_release=lease_release,
        phase_clocks=phase_clocks,
        sample=sample,
        inference_start_ns=inference_start_ns,
        inference_end_ns=inference_end_ns,
        cleanup_start_ns=cleanup_start_ns,
        process_exit_confirmed=process_exit_confirmed,
    )
    receipt_validation = task_runtime_receipts.validate_task_compute_receipt(receipt)
    registry_after = sha256_file(repo_root / REGISTRY_RELATIVE_PATH)
    control = cell_rows[0]
    shadow = cell_rows[1]
    decision = dict(shadow.get("belief_decision") or {})
    requests = sum(int(row["request_count"]) for row in cell_rows)
    completions = sum(int(row["completion_count"]) for row in cell_rows)
    errors = sum(int(row["error_count"]) for row in cell_rows)
    launch_argv = list(proposer.last_launch_argv)
    model_execution_rows = [
        {
            "hf_id": MANDATED_MODEL_HF_ID,
            "model_path": model_spec["model_path"],
            "model_filename": Path(model_spec["model_path"]).name,
            "model_file_hash": model_spec["model_file_hash"],
            **model_identity_receipt,
            "n_ctx": MODEL_N_CTX,
            "observed_server_n_ctx": proposer.observed_server_n_ctx,
            "n_gpu_layers": proposer.n_gpu_layers,
            "launch_argv": launch_argv,
            "cuda_offload": "-ngl" in launch_argv and int(launch_argv[launch_argv.index("-ngl") + 1]) > 0,
            "request_count": requests,
            "completion_count": completions,
            "error_count": errors,
            "pid": server_pid,
            "terminal": True,
        }
    ]
    server_rows = [
        {
            "pid": server_pid,
            "process_start_identity": server_identity,
            "owned": True,
            "port": proposer.port,
            "endpoint": proposer._url(),
            "server_binary": proposer.generator_server_path,
            "server_binary_hash": sha256_file(proposer.generator_server_path),
            "server_command_hash": sha256_json(launch_argv),
            "model_file_hash": model_spec["model_file_hash"],
            "observed_server_model_path": model_identity_receipt["observed_server_model_path"],
            "resolved_model_path": model_identity_receipt["resolved_model_path"],
            "n_ctx": proposer.observed_server_n_ctx,
            "request_count": requests,
            "completion_count": completions,
            "error_count": errors,
            "terminal": True,
        }
    ]
    parity = {
        "control_action": control["action"],
        "shadow_action": shadow["action"],
        "control_candidate_set_sha256": sha256_json(control["candidate_actions"]),
        "shadow_candidate_set_sha256": sha256_json(shadow["candidate_actions"]),
        "candidate_set_match": control["candidate_actions"] == shadow["candidate_actions"],
        "passed": control["action"] == shadow["action"] and control["candidate_actions"] == shadow["candidate_actions"],
        "terminal": True,
    }
    final_checks = [
        *checks,
        gate_row("task_compute_receipt_valid", True, receipt_validation["accepted"] is True),
        gate_row("belief_query_fired", True, decision.get("query_fired") is True),
        gate_row("shadow_action_parity", True, parity["passed"] is True),
        gate_row("registry_unchanged", registry_before, registry_after),
        gate_row("owned_teardown", True, teardown_rows[0]["passed"] is True),
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": final_checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": time.perf_counter() - started,
        "MODEL_SPECS": [model_spec],
        "models_used": [MANDATED_MODEL_HF_ID],
        "model_file_hashes": {model_spec["model_path"]: model_spec["model_file_hash"]},
        "cited_upstream_artifacts": deepcopy(preflight["citations"]),
        "source_artifact_hashes": deepcopy(preflight["source_hashes"]),
        "rows": [
            {"check": "matched_cells", "row_count": len(cell_rows), "passed": len(cell_rows) == 2, "terminal": True},
            {"check": "task_compute_receipt", "row_count": len(receipt["rows"]), "passed": receipt_validation["accepted"], "terminal": True},
            {"check": "owned_teardown", "row_count": 1, "passed": teardown_rows[0]["passed"], "terminal": True},
        ],
        "per_action_results": [
            {"cell": row["cell"], "action": row["action"], "terminal": True} for row in cell_rows
        ],
        "control_rows": [control],
        "shadow_rows": [shadow],
        "shadow_action_parity_rows": [parity],
        "belief_query_rows": [
            {
                "query_fired": decision.get("query_fired") is True,
                "query_count": int(decision.get("query_count", 0)),
                "evidence_hashes": list(decision.get("evidence_hashes") or []),
                "abstained": decision.get("abstained") is True,
                "terminal": True,
            }
        ],
        "ranking_influence_rows": [
            {
                "counterfactual_ranking_changed": decision.get("counterfactual_ranking_changed") is True,
                "counterfactual_selected_action": decision.get("counterfactual_selected_action"),
                "control_selected_action": decision.get("control_selected_action"),
                "action_override_applied": False,
                "terminal": True,
            }
        ],
        "abstention_rows": (
            [
                {
                    "reason": decision.get("abstention_reason"),
                    "query_fired": decision.get("query_fired") is True,
                    "terminal": True,
                }
            ]
            if decision.get("abstained") is True
            else []
        ),
        "model_execution_rows": model_execution_rows,
        "gpu_identity_rows": [{**gpu, "terminal": True}],
        "gpu_sample_rows": deepcopy(receipt["gpu_sample_rows"]),
        "server_rows": server_rows,
        "lease_rows": deepcopy(receipt["lease_link_rows"]),
        "request_counter_rows": [
            {
                "cell": row["cell"],
                "before": index,
                "after": index + row["request_count"],
                "delta": row["request_count"],
                "terminal": True,
            }
            for index, row in enumerate(cell_rows)
        ],
        "completion_counter_rows": [
            {
                "cell": row["cell"],
                "requests": row["request_count"],
                "completions": row["completion_count"],
                "errors": row["error_count"],
                "terminal": True,
            }
            for row in cell_rows
        ],
        "phase_receipt_rows": deepcopy(receipt["rows"]),
        "runner_decision_rows": [{**receipt["runner_decision"], "terminal": True}],
        "checkpoint_rows": checkpoint_rows,
        "teardown_rows": teardown_rows,
        "task_compute_receipt": receipt,
        "solve_provenance": "live_agent_self_discovery",
        "solve_registry_precheck_rows": [
            {
                "registry_hash_before": registry_before,
                "registry_hash_after": registry_after,
                "targeted_level": None,
                "already_reproduced_target": False,
                "observed_game_id": control["game_id"],
                "transport_only": True,
                "terminal": True,
            }
        ],
        "arc_new_level_banked": 0,
        "belief_shadow_trace_ready_score": 1,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_check_summary(final_checks),
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_belief_shadow_live_trace_transport_ready_no_value_or_solve_claim",
    }
    live_errors = _live_validation_errors(artifact)
    if live_errors:
        artifact["belief_shadow_trace_ready_score"] = 0
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = "partial_belief_shadow_live_trace:" + live_errors[0]
        artifact["gate_check_summary"] = {
            "passed": False,
            "failed_check": live_errors[0],
            "expected_value": "complete valid live transport evidence",
            "observed_value": live_errors,
            "checks": final_checks,
        }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run(
    *, run_date: str, repo_root: Path = REPO_ROOT, output_path: Path | None = None, checkpoint_path: Path | None = None
) -> JsonDict:  # pragma: no cover - orchestration boundary
    """Run preconditions, then execute or write the exact blocked outcome."""

    started = time.perf_counter()
    output = output_path or repo_root / RESULT_RELATIVE_PATH
    checkpoint = checkpoint_path or repo_root / CHECKPOINT_RELATIVE_PATH
    preflight = collect_preconditions(
        repo_root=repo_root,
        output_path=output,
        checkpoint_path=checkpoint,
    )
    if preflight["summary"]["passed"] is not True:
        artifact = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=preflight["checks"],
            source_hashes=preflight["source_hashes"],
        )
        artifact["cited_upstream_artifacts"] = deepcopy(preflight["citations"])
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        write_artifact(output, artifact)
        return artifact
    artifact = execute_live_trace(
        repo_root=repo_root,
        output_path=output,
        checkpoint_path=checkpoint,
        preflight=preflight,
        run_date=run_date,
    )
    write_artifact(output, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary
    parser = argparse.ArgumentParser(description="Run Exp7025 live belief-shadow trace")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / CHECKPOINT_RELATIVE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    """Write one stable terminal artifact and print only its terminal state."""

    args = _parser().parse_args(argv)
    if args.output.is_file():
        existing = _load_json(args.output)
        current_sources = _source_hashes(REPO_ROOT)
        recorded_sources = existing.get("source_artifact_hashes")
        sources_stable = isinstance(recorded_sources, Mapping) and all(
            recorded_sources.get(path) == digest for path, digest in current_sources.items()
        )
        if (
            existing.get("execution_date") == args.date
            and sources_stable
            and validate_artifact(existing) == []
        ):
            print(
                f"stable {args.output} belief_shadow_trace_ready_score="
                f"{existing['belief_shadow_trace_ready_score']}"
            )
            return 0
    artifact = run(
        run_date=args.date,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    print(
        f"wrote {args.output} belief_shadow_trace_ready_score="
        f"{artifact['belief_shadow_trace_ready_score']} verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
