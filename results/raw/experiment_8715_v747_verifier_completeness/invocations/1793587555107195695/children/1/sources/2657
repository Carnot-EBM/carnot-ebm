"""Run the repaired, provenance-complete ARC belief-shadow cell.

The audited Exp7025 live runner owns the server, leases, policy construction,
and one official action. This module adds the Exp7032 upstream pins and the
required receipt projections without changing the shared identity bridge.

Spec: REQ-ARC-7032 and SCENARIO-ARC-7032-*.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import task_runtime_receipts
from carnot.agentic import arc_belief_shadow_live_trace as live_core
from carnot.agentic.arc_eval_provenance import (
    build_arc_model_identity_receipt,
    validate_arc_evaluation_row,
)


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7032
SCHEMA = "carnot.exp7032.repaired_belief_shadow_live_trace.v1"
RUN_DATE = "20260905"
RANDOM_SEED = 7_032_202_609_05
INFERENCE_SUBSTRATE = "live_llm_inference"
MANDATED_MODEL_HF_ID = live_core.MANDATED_MODEL_HF_ID
MANDATED_MODEL_NAME = live_core.MANDATED_MODEL_NAME
PRIOR_IDENTITY_FAILURE_VERDICT = "blocked_belief_shadow_live_trace:live_trace_execution"
RESULT_RELATIVE_PATH = Path("results/experiment_7032_repaired_belief_shadow_live_trace.json")
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_7032/checkpoint.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-belief-shadow-live-trace/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7032_repaired_belief_shadow_live_trace.py")
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_7032_repaired_belief_shadow_live_trace.py"
)
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_7032_repaired_belief_shadow_live_trace.py")
REPO_ROOT = Path(__file__).resolve().parents[2]

UPSTREAMS = (
    (
        7030,
        Path("results/experiment_7030_arc_gguf_model_identity_bridge.json"),
        "arc_model_identity_bridge_ready_score",
        "carnot.experiment_7030_arc_gguf_model_identity_bridge",
    ),
    (
        7031,
        Path("results/experiment_7031_arc_model_identity_cold_audit.json"),
        "arc_model_identity_audit_ready_score",
        "carnot.experiment_7031_arc_model_identity_cold_audit",
    ),
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "upstream_gate_rows",
    "MODEL_SPECS",
    "model_specs",
    "models_used",
    "requested_model_path",
    "requested_model_filename",
    "requested_hf_id",
    "requested_revision",
    "observed_server_model_path",
    "resolved_model_path",
    "model_file_hashes",
    "quantization",
    "tokenizer_receipt",
    "no_autotokenizer_used",
    "server_process_rows",
    "port_lease_rows",
    "gpu_lease_rows",
    "gpu_sample_rows",
    "cuda_layer_offload_confirmed",
    "context_receipt_rows",
    "phase_receipt_rows",
    "rows",
    "per_decision_rows",
    "action_parity_rows",
    "belief_query_rows",
    "observation_rows",
    "progress_rows",
    "cleanup_rows",
    "solve_registry_precheck_rows",
    "solve_provenance",
    "live_model_invoked",
    "belief_shadow_trace_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why the transport result needs it.",
    "preconditions_checked": "Exact gates prevent fallback after a missing live resource.",
    "inference_substrate": "The declared substrate subjects this live model run to the correct audit floor.",
    "duration_s": "Measured wall time makes real model load and inference falsifiable.",
    "source_artifact_hashes": "Source hashes bind this result to the code and registry it used.",
    "cited_upstream_artifacts": "Artifact citations identify the exact prerequisite evidence.",
    "upstream_gate_rows": "Gate rows prove both repaired identity experiments remained clean.",
    "MODEL_SPECS": "The uppercase field preserves the experiment model-selection contract.",
    "model_specs": "The lowercase field lets standard methodology audits identify the executed model.",
    "models_used": "The invoked-model list prevents a legacy model from satisfying the task.",
    "requested_model_path": "The snapshot path preserves the human-requested GGUF identity.",
    "requested_model_filename": "The requested filename keeps the GGUF schema separate from its blob.",
    "requested_hf_id": "The hub ID prevents a same-name model from another repository.",
    "requested_revision": "The revision binds the request to one immutable snapshot.",
    "observed_server_model_path": "The observed path records what the server actually loaded.",
    "resolved_model_path": "The resolved path joins the snapshot link to its content blob.",
    "model_file_hashes": "Content hashes prove requested and observed paths contain the same bytes.",
    "quantization": "Quantization affects memory use and output, so it is part of model identity.",
    "tokenizer_receipt": "The embedded-tokenizer receipt prevents an invalid Transformers load claim.",
    "no_autotokenizer_used": "True proves the GGUF-only repository never entered AutoTokenizer.",
    "server_process_rows": "Process rows bind command, PID, port, model, context, and GPU.",
    "port_lease_rows": "Port lease rows prove the endpoint was owned and later released.",
    "gpu_lease_rows": "GPU lease rows prove the selected device belonged to this task.",
    "gpu_sample_rows": "In-phase samples prove the model process used the leased GPU.",
    "cuda_layer_offload_confirmed": "Observed positive layer offload rules out a CPU headline fallback.",
    "context_receipt_rows": "Context rows compare the requested and server-observed token windows.",
    "phase_receipt_rows": "Phase clocks attribute setup, load, inference, write, and cleanup work.",
    "rows": "Summary rows keep each readiness family independently countable.",
    "per_decision_rows": "Decision rows preserve prompts, rankings, queries, actions, and model counts.",
    "action_parity_rows": "Parity rows prove belief did not change the emitted action.",
    "belief_query_rows": "Query rows distinguish an active belief path from a silent no-op.",
    "observation_rows": "Observation hashes prove both decisions saw the same pre-action state.",
    "progress_rows": "Progress rows separate transport evidence from any incidental level change.",
    "cleanup_rows": "Cleanup rows prove only owned resources were stopped and released.",
    "solve_registry_precheck_rows": "Registry hashes prove no reproduced solve was targeted or changed.",
    "solve_provenance": "The enum labels the official action source without making a solve claim.",
    "live_model_invoked": "True requires completed requests to the owned model server.",
    "belief_shadow_trace_ready_score": "One requires every identity, parity, compute, and cleanup gate.",
    "random_seed": "One seed fixes model sampling and policy order across both cells.",
    "reproducibility_checksum": "A canonical digest detects any later artifact change.",
    "gate_check_summary": "The first exact failure makes a blocked run actionable.",
    "verifier_is_oracle": "False states that this transport check does not define game correctness.",
    "verdict_class": "A closed class separates readiness from blocked or partial execution.",
    "honest_verdict": "A class-consistent prefix gives downstream tools one terminal meaning.",
}

sha256_file = live_core.sha256_file
sha256_json = live_core.sha256_json
gate_row = live_core.gate_row
gate_check_summary = live_core.gate_check_summary
resolve_model_spec = live_core.resolve_model_spec


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact except its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def extract_quantization(filename: Any) -> str | None:
    """Read the known quantization token from a requested GGUF filename."""

    name = str(filename or "").upper()
    for token in (
        "IQ4_XS",
        "IQ3_XXS",
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q5_K_S",
        "Q4_K_M",
        "Q4_K_S",
        "Q3_K_M",
        "Q2_K",
    ):
        if token in name:
            return token
    return None


def pinned_source_hash_checks(
    root: Path, artifact: Mapping[str, Any], experiment_id: int
) -> list[JsonDict]:
    """Recompute every source hash cited by one upstream identity artifact."""

    recorded = artifact.get("source_artifact_hashes")
    if not isinstance(recorded, Mapping):
        return [gate_row(f"exp{experiment_id}_source_hashes", "mapping", type(recorded).__name__)]
    rows = []
    for relative, expected in sorted(recorded.items(), key=lambda item: str(item[0])):
        path = root / str(relative)
        observed = sha256_file(path) if path.is_file() else "missing"
        rows.append(gate_row(f"exp{experiment_id}_source_hash:{relative}", expected, observed))
    return rows


def observation_receipt(frame: Any, *, cell: str, position: str) -> JsonDict:
    """Hash one immutable frame view and retain only audit-safe metadata."""

    if hasattr(frame, "model_dump"):
        payload = frame.model_dump(mode="json")
    elif isinstance(frame, Mapping):
        payload = deepcopy(dict(frame))
    else:
        payload = {"representation": str(frame)}
    frame_value = payload.get("frame") if isinstance(payload, Mapping) else None
    state = payload.get("state") if isinstance(payload, Mapping) else None
    if hasattr(state, "value"):
        state = state.value
    return {
        "cell": str(cell),
        "position": str(position),
        "observation_sha256": sha256_json(payload),
        "frame_sha256": sha256_json(frame_value),
        "game_id": payload.get("game_id") if isinstance(payload, Mapping) else None,
        "guid": payload.get("guid") if isinstance(payload, Mapping) else None,
        "state": state,
        "levels_completed": payload.get("levels_completed")
        if isinstance(payload, Mapping)
        else None,
        "win_levels": payload.get("win_levels") if isinstance(payload, Mapping) else None,
        "available_actions": list(payload.get("available_actions") or [])
        if isinstance(payload, Mapping)
        else [],
        "terminal": True,
    }


def _load_json(path: Path) -> JsonDict:  # pragma: no cover - filesystem precondition
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - filesystem precondition
    paths = (
        MODULE_RELATIVE_PATH,
        SCRIPT_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
        Path("python/carnot/agentic/arc_belief_shadow_live_trace.py"),
        Path("python/carnot/agentic/arc_eval_provenance.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("ops/arc_solve_registry.yaml"),
        Path("results/experiment_7017_task_linked_compute_receipts.json"),
        Path("results/experiment_7024_belief_aware_e3_selector.json"),
        Path("results/experiment_7025_belief_shadow_live_trace.json"),
        *(relative for _number, relative, _field, _module in UPSTREAMS),
    )
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in paths
        if (root / relative).is_file()
    }


def _validate_upstreams(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover - host precondition
    """Validate both identity artifacts and all evidence that they pin."""

    import importlib

    checks: list[JsonDict] = []
    citations: list[JsonDict] = []
    for number, relative, ready_field, module_name in UPSTREAMS:
        path = root / relative
        artifact = _load_json(path)
        checks.append(gate_row(f"exp{number}_artifact_readable", True, bool(artifact)))
        checks.append(gate_row(f"exp{number}.{ready_field}", 1, artifact.get(ready_field)))
        try:
            validator = importlib.import_module(module_name).validate_artifact
            validation = validator(artifact)
        except Exception as exc:  # noqa: BLE001 - exact failure belongs in the gate row
            validation = [f"{type(exc).__name__}: {exc}"]
        checks.append(gate_row(f"exp{number}_artifact_valid", [], validation))
        checks.extend(pinned_source_hash_checks(root, artifact, number))
        artifact_hash = sha256_file(path) if path.is_file() else "missing"
        citations.append(
            {
                "experiment_id": number,
                "fields_imported": [ready_field, "source_artifact_hashes"],
                "sha256": artifact_hash,
                "terminal": True,
            }
        )
        for row in artifact.get("cited_upstream_artifacts", []):
            if not isinstance(row, Mapping) or not row.get("path") or not row.get("artifact_hash"):
                continue
            cited_path = root / str(row["path"])
            observed = sha256_file(cited_path) if cited_path.is_file() else "missing"
            checks.append(
                gate_row(
                    f"exp{number}_cited_artifact_hash:{row['path']}",
                    row["artifact_hash"],
                    observed,
                )
            )
    return checks, citations


def collect_preconditions(
    *, repo_root: Path, output_path: Path, checkpoint_path: Path
) -> JsonDict:  # pragma: no cover - official live host boundary
    """Add pinned identity gates before the audited live runner can start a model."""

    upstream_checks, citations = _validate_upstreams(repo_root)
    core = live_core.collect_preconditions(
        repo_root=repo_root,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    checks = [*upstream_checks, *core["checks"]]
    return {
        **core,
        "checks": checks,
        "summary": gate_check_summary(checks),
        "identity_citations": citations,
        "source_hashes": _source_hashes(repo_root),
        "upstream_gate_rows": deepcopy(upstream_checks),
    }


@contextmanager
def _capture_live_telemetry() -> Any:  # pragma: no cover - official SDK boundary
    """Observe model and frame data without changing any policy return value."""

    from carnot.agentic import arc_competition_agent as competition
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    telemetry: JsonDict = {"model_call_rows": [], "observation_rows": []}
    original_complete = LocalGGUFProposer.complete_text
    original_factory = competition.make_carnot_agent

    def observed_complete(self: Any, prompt: str, **kwargs: Any) -> tuple[bool, str]:
        cell_index = len(telemetry["model_call_rows"])
        cell = (
            live_core.CELL_IDS[cell_index]
            if cell_index < len(live_core.CELL_IDS)
            else f"call_{cell_index}"
        )
        started_ns = time.monotonic_ns()
        result = original_complete(self, prompt, **kwargs)
        telemetry["model_call_rows"].append(
            {
                "cell": cell,
                "prompt_sha256": sha256_json(prompt),
                "requested_tokens": int(kwargs.get("max_tokens") or self.max_tokens),
                "generated_tokens": int(getattr(self, "last_generated_tokens", -1)),
                "request_start_ns": started_ns,
                "request_end_ns": time.monotonic_ns(),
                "completed": result[0] is True,
                "response_sha256": sha256_json(result[1]),
                "terminal": True,
            }
        )
        return result

    def observed_factory(*args: Any, **kwargs: Any) -> type:
        agent_class = original_factory(*args, **kwargs)
        cell = "belief_shadow" if kwargs.get("belief_aware_selector") is True else "control"

        class ObservedAgent(agent_class):  # type: ignore[misc, valid-type]
            def choose_action(self, frames: Any, latest_frame: Any) -> Any:
                try:
                    telemetry["observation_rows"].append(
                        observation_receipt(latest_frame, cell=cell, position="pre_action")
                    )
                except Exception as exc:  # noqa: BLE001 - observation must not alter action selection
                    telemetry.setdefault("capture_errors", []).append(
                        f"pre:{cell}:{type(exc).__name__}:{exc}"
                    )
                return super().choose_action(frames, latest_frame)

            def take_action(self, action: Any) -> Any:
                returned = super().take_action(action)
                try:
                    telemetry["observation_rows"].append(
                        observation_receipt(returned, cell=cell, position="post_action")
                    )
                except Exception as exc:  # noqa: BLE001 - observation must not alter environment return
                    telemetry.setdefault("capture_errors", []).append(
                        f"post:{cell}:{type(exc).__name__}:{exc}"
                    )
                return returned

        return ObservedAgent

    LocalGGUFProposer.complete_text = observed_complete
    competition.make_carnot_agent = observed_factory
    try:
        yield telemetry
    finally:
        LocalGGUFProposer.complete_text = original_complete
        competition.make_carnot_agent = original_factory


def _rebuild_task_receipt(
    core: Mapping[str, Any], *, run_date: str, server: Mapping[str, Any], spec: Mapping[str, Any]
) -> JsonDict:
    """Re-seal the shared receipt with the actual Exp7032 command and task ID."""

    old = core.get("task_compute_receipt")
    if not isinstance(old, Mapping) or not old:
        return {}
    task_id = f"exp7032-repaired-belief-shadow-live-trace:{run_date}:{os.getpid()}"
    gpu_rows = core.get("gpu_identity_rows")
    gpu = gpu_rows[0] if isinstance(gpu_rows, list) and gpu_rows else {}
    device = f"cuda:{gpu.get('index', 0)}"
    lease_links = []
    for value in old.get("lease_link_rows", []):
        lease_links.append(
            {**dict(value), "task_id": task_id, "gpu_uuid": gpu.get("gpu_uuid"), "device": device}
        )
    model_rows = []
    for value in old.get("model_process_rows", []):
        model_rows.append(
            {
                **dict(value),
                "task_id": task_id,
                "pid": server.get("pid"),
                "model_id": MANDATED_MODEL_HF_ID,
                "model_file_hash": spec.get("model_file_hash"),
                "gpu_uuid": gpu.get("gpu_uuid"),
                "device": device,
            }
        )
    samples = []
    for value in old.get("gpu_sample_rows", []):
        samples.append(
            {
                **dict(value),
                "task_id": task_id,
                "pid": server.get("pid"),
                "model_id": MANDATED_MODEL_HF_ID,
                "model_file_hash": spec.get("model_file_hash"),
                "gpu_uuid": gpu.get("gpu_uuid"),
                "device_uuid": gpu.get("gpu_uuid"),
                "device": device,
            }
        )
    cleanups = []
    for value in old.get("cleanup_rows", []):
        row = {**dict(value), "task_id": task_id}
        if row.get("kind") == "model":
            row.update({"pid": server.get("pid"), "model_id": MANDATED_MODEL_HF_ID})
        cleanups.append(row)
    clocks = [
        {
            "phase": row.get("phase"),
            "monotonic_start_ns": row.get("monotonic_start_ns"),
            "monotonic_end_ns": row.get("monotonic_end_ns"),
            "wall_clock_start": row.get("wall_clock_start"),
            "wall_clock_end": row.get("wall_clock_end"),
        }
        for row in old.get("rows", [])
    ]
    return task_runtime_receipts.build_task_compute_receipt(
        task_id=task_id,
        task_process_identity=dict(old.get("task_process_identity") or {}),
        lease_link_rows=lease_links,
        phase_clocks=clocks,
        gpu_sample_rows=samples,
        model_process_rows=model_rows,
        runner_decision=dict(old.get("runner_decision") or {}),
        cleanup_rows=cleanups,
        command=[
            os.environ.get("PYTHON", ".venv/bin/python"),
            SCRIPT_RELATIVE_PATH.as_posix(),
            "--date",
            run_date,
        ],
        config={
            "random_seed": RANDOM_SEED,
            "action_budget": live_core.ACTION_BUDGET,
            "prompt_sha256": sha256_json(live_core.MODEL_PROMPT),
            "max_tokens": live_core.MODEL_MAX_TOKENS,
            "temperature": live_core.MODEL_TEMPERATURE,
            "n_ctx": live_core.MODEL_N_CTX,
        },
    )


def _empty_fields() -> JsonDict:
    """Create every required evidence field before any live resource exists."""

    return {
        "upstream_gate_rows": [],
        "MODEL_SPECS": [],
        "model_specs": [],
        "models_used": [],
        "requested_model_path": None,
        "requested_model_filename": None,
        "requested_hf_id": None,
        "requested_revision": None,
        "observed_server_model_path": None,
        "resolved_model_path": None,
        "model_file_hashes": {},
        "quantization": None,
        "tokenizer_receipt": {
            "source": "embedded_gguf",
            "validated_by": "not_started",
            "autotokenizer_used": False,
            "terminal": True,
        },
        "no_autotokenizer_used": True,
        "server_process_rows": [],
        "port_lease_rows": [],
        "gpu_lease_rows": [],
        "gpu_sample_rows": [],
        "cuda_layer_offload_confirmed": False,
        "context_receipt_rows": [],
        "phase_receipt_rows": [],
        "rows": [],
        "per_decision_rows": [],
        "action_parity_rows": [],
        "belief_query_rows": [],
        "observation_rows": [],
        "progress_rows": [],
        "cleanup_rows": [],
        "solve_registry_precheck_rows": [],
    }


def _is_repeated_identity_failure(summary: Mapping[str, Any]) -> bool:
    observed = str(summary.get("observed_value", "")).lower()
    return summary.get("failed_check") == "live_trace_execution" and any(
        token in observed
        for token in ("model identity", "model_filename", "requested_model_filename")
    )


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a complete terminal blocker without inferred live evidence."""

    summary = gate_check_summary(preconditions)
    failed = summary.get("failed_check") or "unknown_precondition"
    verdict = (
        PRIOR_IDENTITY_FAILURE_VERDICT
        if _is_repeated_identity_failure(summary)
        else f"blocked_repaired_belief_shadow_live_trace:{failed}"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": str(run_date),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": dict(source_artifact_hashes),
        "cited_upstream_artifacts": [dict(row) for row in cited_upstream_artifacts],
        **_empty_fields(),
        "solve_provenance": "live_agent_self_discovery",
        "live_model_invoked": False,
        "belief_shadow_trace_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": verdict,
        "game_level_solve_claim": False,
    }
    artifact["upstream_gate_rows"] = [
        dict(row)
        for row in preconditions
        if str(row.get("check", "")).startswith(("exp7030", "exp7031"))
    ]
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _decision_rows(core: Mapping[str, Any], telemetry: Mapping[str, Any]) -> list[JsonDict]:
    """Join policy decisions to their model and pre-action observation receipts."""

    calls = {str(row.get("cell")): dict(row) for row in telemetry.get("model_call_rows", [])}
    observations = {
        str(row.get("cell")): dict(row)
        for row in telemetry.get("observation_rows", [])
        if row.get("position") == "pre_action"
    }
    rows = []
    for cell, source_rows in (
        ("control", core.get("control_rows", [])),
        ("belief_shadow", core.get("shadow_rows", [])),
    ):
        if not isinstance(source_rows, list) or not source_rows:
            continue
        source = dict(source_rows[0])
        decision = dict(source.get("belief_decision") or {})
        base = list(decision.get("base_actions") or source.get("candidate_actions") or [])
        shadow = list(decision.get("ranked_actions") or base)
        call = calls.get(cell, {})
        rows.append(
            {
                "cell": cell,
                "game_id": source.get("game_id"),
                "episode_guid": source.get("episode_guid"),
                "pre_action_observation_sha256": observations.get(cell, {}).get(
                    "observation_sha256"
                ),
                "prompt_sha256": source.get("prompt_hash") or call.get("prompt_sha256"),
                "belief_evidence_ids": list(decision.get("evidence_hashes") or []),
                "query_result": {
                    "query_fired": decision.get("query_fired") is True,
                    "query_count": int(decision.get("query_count", 0)),
                    "abstained": decision.get("abstained") is True,
                    "abstention_reason": decision.get("abstention_reason"),
                },
                "base_ranking": base,
                "shadow_ranking": shadow,
                "selected_action": deepcopy(source.get("action")),
                "model_calls": int(source.get("request_count", 0)),
                "model_completions": int(source.get("completion_count", 0)),
                "model_errors": int(source.get("error_count", 0)),
                "model_tokens_requested": call.get("requested_tokens"),
                "model_tokens_generated": call.get("generated_tokens"),
                "arc_eval_provenance": deepcopy(source.get("arc_eval_provenance")),
                "solve_provenance": source.get("solve_provenance"),
                "terminal": True,
            }
        )
    return rows


def adapt_core_artifact(
    core: Mapping[str, Any],
    *,
    telemetry: Mapping[str, Any],
    run_date: str,
    checkpoint_path: Path,
    upstream_gate_rows: Sequence[Mapping[str, Any]],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
) -> JsonDict:
    """Project the audited live-core result into the complete Exp7032 schema."""

    if core.get("belief_shadow_trace_ready_score") != 1:
        blocked = build_blocked_artifact(
            run_date=run_date,
            duration_s=float(core.get("duration_s", 0.0)),
            preconditions=list(core.get("preconditions_checked") or upstream_gate_rows),
            source_artifact_hashes=source_artifact_hashes,
            cited_upstream_artifacts=cited_upstream_artifacts,
        )
        blocked["MODEL_SPECS"] = deepcopy(list(core.get("MODEL_SPECS") or []))
        blocked["model_specs"] = deepcopy(blocked["MODEL_SPECS"])
        blocked["models_used"] = deepcopy(list(core.get("models_used") or []))
        blocked["model_file_hashes"] = deepcopy(dict(core.get("model_file_hashes") or {}))
        blocked["cleanup_rows"] = deepcopy(list(core.get("teardown_rows") or []))
        blocked["solve_registry_precheck_rows"] = deepcopy(
            list(core.get("solve_registry_precheck_rows") or [])
        )
        blocked["live_model_invoked"] = bool(blocked["models_used"])
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        return blocked

    specs = list(core.get("MODEL_SPECS") or [])
    spec = dict(specs[0])
    executions = list(core.get("model_execution_rows") or [])
    execution = dict(executions[0])
    servers = list(core.get("server_rows") or [])
    server = dict(servers[0])
    identity = {
        key: execution.get(key)
        for key in (
            "requested_model_path",
            "requested_model_filename",
            "requested_hf_id",
            "requested_revision",
            "observed_server_model_path",
            "resolved_model_path",
            "model_file_hash",
        )
    }
    receipt = _rebuild_task_receipt(core, run_date=run_date, server=server, spec=spec)
    server_process = {
        **server,
        "server_command": deepcopy(execution.get("launch_argv")),
        "gpu_uuid": (core.get("gpu_identity_rows") or [{}])[0].get("gpu_uuid"),
        "cuda_layer_offload_confirmed": execution.get("cuda_offload") is True,
        "terminal": True,
    }
    teardown = list(core.get("teardown_rows") or [])
    port_lease_rows = [
        {
            "port": server.get("port"),
            "endpoint": server.get("endpoint"),
            "owner_pid": server.get("pid"),
            "lease_path": str(
                checkpoint_path.parent / "port-leases" / f"port-{server.get('port')}.lease"
            ),
            "owned": server.get("owned") is True,
            "released": bool(teardown and teardown[0].get("port_released") is True),
            "terminal": True,
        }
    ]
    context_rows = [
        {
            "requested_n_ctx": execution.get("n_ctx"),
            "observed_server_n_ctx": execution.get("observed_server_n_ctx"),
            "passed": execution.get("n_ctx") == execution.get("observed_server_n_ctx"),
            "terminal": True,
        }
    ]
    observation_rows = deepcopy(list(telemetry.get("observation_rows") or []))
    before = next(
        (
            row
            for row in observation_rows
            if row.get("cell") == "control" and row.get("position") == "pre_action"
        ),
        {},
    )
    after = next(
        (
            row
            for row in observation_rows
            if row.get("cell") == "control" and row.get("position") == "post_action"
        ),
        {},
    )
    level_before = before.get("levels_completed")
    level_after = after.get("levels_completed")
    level_delta = (
        level_after - level_before
        if isinstance(level_before, int) and isinstance(level_after, int)
        else None
    )
    progress_rows = [
        {
            "progress_type": "transport_only",
            "credited_action_count": 1,
            "level_before": level_before,
            "level_after": level_after,
            "level_delta": level_delta,
            "incidental_level_completion": bool(level_delta and level_delta > 0),
            "headline_solve_claim": False,
            "terminal": True,
        }
    ]
    combined_citations = [dict(row) for row in cited_upstream_artifacts]
    combined_citations.extend(
        dict(row)
        for row in core.get("cited_upstream_artifacts", [])
        if dict(row) not in combined_citations
    )
    all_checks = [*list(core.get("preconditions_checked") or []), *list(upstream_gate_rows)]
    action_parity = deepcopy(list(core.get("shadow_action_parity_rows") or []))
    decisions = _decision_rows(core, telemetry)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": str(run_date),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(all_checks),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(core.get("duration_s", 0.0)),
        "source_artifact_hashes": dict(source_artifact_hashes),
        "cited_upstream_artifacts": combined_citations,
        "upstream_gate_rows": [dict(row) for row in upstream_gate_rows],
        "MODEL_SPECS": deepcopy(specs),
        "model_specs": deepcopy(specs),
        "models_used": [MANDATED_MODEL_HF_ID],
        "requested_model_path": identity["requested_model_path"],
        "requested_model_filename": identity["requested_model_filename"],
        "requested_hf_id": identity["requested_hf_id"],
        "requested_revision": identity["requested_revision"],
        "observed_server_model_path": identity["observed_server_model_path"],
        "resolved_model_path": identity["resolved_model_path"],
        "model_file_hashes": deepcopy(dict(core.get("model_file_hashes") or {})),
        "quantization": extract_quantization(identity["requested_model_filename"]),
        "tokenizer_receipt": {
            "source": "embedded_gguf",
            "requested_model_filename": identity["requested_model_filename"],
            "validated_by": "owned_cuda_llama_server",
            "autotokenizer_used": False,
            "terminal": True,
        },
        "no_autotokenizer_used": True,
        "server_process_rows": [server_process],
        "port_lease_rows": port_lease_rows,
        "gpu_lease_rows": deepcopy(
            list(receipt.get("lease_link_rows") or core.get("lease_rows") or [])
        ),
        "gpu_sample_rows": deepcopy(
            list(receipt.get("gpu_sample_rows") or core.get("gpu_sample_rows") or [])
        ),
        "cuda_layer_offload_confirmed": execution.get("cuda_offload") is True,
        "context_receipt_rows": context_rows,
        "phase_receipt_rows": deepcopy(
            list(receipt.get("rows") or core.get("phase_receipt_rows") or [])
        ),
        "rows": deepcopy(list(core.get("rows") or [])),
        "per_decision_rows": decisions,
        "action_parity_rows": action_parity,
        "belief_query_rows": deepcopy(list(core.get("belief_query_rows") or [])),
        "observation_rows": observation_rows,
        "progress_rows": progress_rows,
        "cleanup_rows": deepcopy(teardown),
        "solve_registry_precheck_rows": deepcopy(
            list(core.get("solve_registry_precheck_rows") or [])
        ),
        "solve_provenance": core.get("solve_provenance"),
        "live_model_invoked": sum(int(row.get("model_completions", 0)) for row in decisions) == 2,
        "belief_shadow_trace_ready_score": 1,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_check_summary(all_checks),
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_repaired_belief_shadow_live_trace_transport_ready_no_solve_claim",
        "task_compute_receipt": receipt,
        "game_level_solve_claim": False,
        "arc_new_level_banked": 0,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _positive_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute every fact needed for a positive transport verdict."""

    errors: list[str] = []
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 60.0:
        errors.append("duration_live_llm_floor_invalid")
    upstream = artifact.get("upstream_gate_rows")
    if (
        not isinstance(upstream, list)
        or not upstream
        or any(row.get("passed") is not True for row in upstream)
    ):
        errors.append("upstream_gate_rows_invalid")
    preconditions = artifact.get("preconditions_checked")
    if (
        not isinstance(preconditions, list)
        or not preconditions
        or any(row.get("passed") is not True for row in preconditions)
    ):
        errors.append("preconditions_invalid")
    specs = artifact.get("MODEL_SPECS")
    spec = specs[0] if isinstance(specs, list) and len(specs) == 1 else {}
    if (
        not isinstance(spec, Mapping)
        or spec.get("hf_id") != MANDATED_MODEL_HF_ID
        or spec.get("resolved_via") != "cached_sota_pair"
        or artifact.get("model_specs") != specs
        or artifact.get("models_used") != [MANDATED_MODEL_HF_ID]
    ):
        errors.append("model_specs_invalid")
    identity_spec = {
        "model_path": artifact.get("requested_model_path"),
        "model_filename": artifact.get("requested_model_filename"),
        "hf_id": artifact.get("requested_hf_id"),
        "revision": artifact.get("requested_revision"),
        "model_file_hash": spec.get("model_file_hash") if isinstance(spec, Mapping) else None,
    }
    try:
        identity = build_arc_model_identity_receipt(
            selected_model_spec=identity_spec,
            observed_server_model_path=artifact.get("observed_server_model_path"),
        )
    except (TypeError, ValueError):
        errors.append("model_identity_invalid")
    else:
        if artifact.get("resolved_model_path") != identity["resolved_model_path"]:
            errors.append("model_identity_resolved_path_invalid")
        hashes = artifact.get("model_file_hashes")
        if (
            not isinstance(hashes, Mapping)
            or hashes.get(identity["requested_model_path"]) != identity["model_file_hash"]
        ):
            errors.append("model_file_hashes_invalid")
    if artifact.get("quantization") is None:
        errors.append("quantization_invalid")
    tokenizer = artifact.get("tokenizer_receipt")
    if (
        artifact.get("no_autotokenizer_used") is not True
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("source") != "embedded_gguf"
        or tokenizer.get("autotokenizer_used") is not False
    ):
        errors.append("tokenizer_receipt_invalid")
    servers = artifact.get("server_process_rows")
    server = servers[0] if isinstance(servers, list) and len(servers) == 1 else {}
    command = server.get("server_command") if isinstance(server, Mapping) else None
    if (
        not isinstance(command, list)
        or "-ngl" not in command
        or not isinstance(server.get("pid"), int)
        or not isinstance(server.get("port"), int)
    ):
        errors.append("server_process_rows_invalid")
    if (
        artifact.get("cuda_layer_offload_confirmed") is not True
        or server.get("cuda_layer_offload_confirmed") is not True
    ):
        errors.append("cuda_offload_invalid")
    context = artifact.get("context_receipt_rows")
    if (
        not isinstance(context, list)
        or not context
        or any(row.get("passed") is not True for row in context)
    ):
        errors.append("context_receipt_rows_invalid")
    decisions = artifact.get("per_decision_rows")
    decisions = decisions if isinstance(decisions, list) else []
    if len(decisions) != 2 or [row.get("cell") for row in decisions] != [
        "control",
        "belief_shadow",
    ]:
        errors.append("per_decision_rows_invalid")
    else:
        if decisions[0].get("selected_action") != decisions[1].get("selected_action"):
            errors.append("action_parity_decisions_invalid")
        if decisions[0].get("pre_action_observation_sha256") != decisions[1].get(
            "pre_action_observation_sha256"
        ):
            errors.append("pre_action_state_mismatch")
        if any(
            row.get("model_calls") != 1
            or row.get("model_completions") != 1
            or row.get("model_errors") != 0
            for row in decisions
        ):
            errors.append("model_call_counts_invalid")
        if any(
            not isinstance(row.get("model_tokens_generated"), int)
            or row.get("model_tokens_generated") < 0
            for row in decisions
        ):
            errors.append("model_token_counts_invalid")
        for row in decisions:
            decision = validate_arc_evaluation_row(
                {
                    "arc_eval_provenance": row.get("arc_eval_provenance"),
                    "solve_provenance": row.get("solve_provenance"),
                }
            )
            if not decision.valid:
                errors.append("arc_eval_provenance_invalid")
    parity = artifact.get("action_parity_rows")
    if (
        not isinstance(parity, list)
        or not parity
        or any(
            row.get("passed") is not True
            or row.get("candidate_set_match") is not True
            or row.get("control_action") != row.get("shadow_action")
            for row in parity
        )
    ):
        errors.append("action_parity_rows_invalid")
    queries = artifact.get("belief_query_rows")
    if (
        not isinstance(queries, list)
        or not queries
        or sum(int(row.get("query_count", 0)) for row in queries if row.get("query_fired") is True)
        <= 0
        or not any(row.get("evidence_hashes") for row in queries)
    ):
        errors.append("belief_query_rows_invalid")
    observations = artifact.get("observation_rows")
    if not isinstance(observations, list) or len(observations) < 3:
        errors.append("observation_rows_invalid")
    progress = artifact.get("progress_rows")
    if (
        not isinstance(progress, list)
        or len(progress) != 1
        or progress[0].get("headline_solve_claim") is not False
    ):
        errors.append("progress_rows_invalid")
    cleanup = artifact.get("cleanup_rows")
    if (
        not isinstance(cleanup, list)
        or not cleanup
        or any(
            row.get("passed") is not True
            or row.get("process_exit_confirmed") is not True
            or row.get("process_reaped") is not True
            or row.get("lease_released") is not True
            or row.get("port_released") is not True
            for row in cleanup
        )
    ):
        errors.append("cleanup_rows_invalid")
    ports = artifact.get("port_lease_rows")
    if (
        not isinstance(ports, list)
        or not ports
        or any(row.get("owned") is not True or row.get("released") is not True for row in ports)
    ):
        errors.append("port_lease_rows_invalid")
    gpu_leases = artifact.get("gpu_lease_rows")
    if (
        not isinstance(gpu_leases, list)
        or not gpu_leases
        or any(row.get("released") is not True for row in gpu_leases)
    ):
        errors.append("gpu_lease_rows_invalid")
    if not artifact.get("gpu_sample_rows"):
        errors.append("gpu_sample_rows_invalid")
    receipt = artifact.get("task_compute_receipt")
    if task_runtime_receipts.validate_task_compute_receipt(receipt).get("accepted") is not True:
        errors.append("task_compute_receipt_invalid")
    if artifact.get("phase_receipt_rows") != (
        receipt.get("rows") if isinstance(receipt, Mapping) else None
    ):
        errors.append("phase_receipt_rows_invalid")
    registry = artifact.get("solve_registry_precheck_rows")
    if (
        not isinstance(registry, list)
        or not registry
        or any(
            row.get("registry_hash_before") != row.get("registry_hash_after")
            or row.get("already_reproduced_target") is not False
            for row in registry
        )
    ):
        errors.append("solve_registry_precheck_rows_invalid")
    if artifact.get("solve_provenance") not in {
        "live_agent_self_discovery",
        "development_proxy",
        "outer_loop_re",
    }:
        errors.append("solve_provenance_invalid")
    if artifact.get("live_model_invoked") is not True:
        errors.append("live_model_invoked_invalid")
    if (
        artifact.get("game_level_solve_claim") is not False
        or artifact.get("arc_new_level_banked") != 0
    ):
        errors.append("game_level_solve_claim_invalid")
    return list(dict.fromkeys(errors))


def validate_artifact(artifact: Any) -> list[str]:
    """Validate schema, terminal semantics, checksum, and positive evidence."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    score = artifact.get("belief_shadow_trace_ready_score")
    if type(score) is not int or score not in (0, 1):
        errors.append("ready_score_invalid")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping) or not {
        "passed",
        "failed_check",
        "expected_value",
        "observed_value",
    } <= set(summary):
        errors.append("gate_check_summary_invalid")
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class == "blocked":
        if (
            score != 0
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or not verdict.startswith("blocked_")
        ):
            errors.append("blocked_terminal_semantics_invalid")
    elif verdict_class == "positive":
        live_errors = _positive_errors(artifact)
        errors.extend(live_errors)
        if (
            score != int(not live_errors)
            or not verdict.startswith("complete_positive_")
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not True
        ):
            errors.append("positive_terminal_semantics_invalid")
    else:
        errors.append("verdict_class_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish the terminal Exp7032 artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp7032 artifact: " + ";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(artifact), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    *,
    run_date: str,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> JsonDict:  # pragma: no cover - official live orchestration
    """Run preflight, one bounded live trace, validation, and atomic output."""

    started = time.perf_counter()
    output = output_path or repo_root / RESULT_RELATIVE_PATH
    checkpoint = checkpoint_path or repo_root / CHECKPOINT_RELATIVE_PATH
    preflight = collect_preconditions(
        repo_root=repo_root,
        output_path=output,
        checkpoint_path=checkpoint,
    )
    citations = preflight["identity_citations"]
    if preflight["summary"]["passed"] is not True:
        artifact = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=preflight["checks"],
            source_artifact_hashes=preflight["source_hashes"],
            cited_upstream_artifacts=citations,
        )
        write_artifact(output, artifact)
        return artifact
    with _capture_live_telemetry() as telemetry:
        core = live_core.execute_live_trace(
            repo_root=repo_root,
            output_path=output,
            checkpoint_path=checkpoint,
            preflight=preflight,
            run_date=run_date,
        )
    artifact = adapt_core_artifact(
        core,
        telemetry=telemetry,
        run_date=run_date,
        checkpoint_path=checkpoint,
        upstream_gate_rows=preflight["upstream_gate_rows"],
        cited_upstream_artifacts=citations,
        source_artifact_hashes=preflight["source_hashes"],
    )
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    if artifact["belief_shadow_trace_ready_score"] == 1:
        _ = validate_artifact(artifact)
        artifact["duration_s"] = time.perf_counter() - started
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        checks = [*preflight["checks"], gate_row("artifact_validation", [], errors)]
        artifact = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=checks,
            source_artifact_hashes=preflight["source_hashes"],
            cited_upstream_artifacts=citations,
        )
    write_artifact(output, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary
    parser = argparse.ArgumentParser(description="Run Exp7032 repaired belief-shadow trace")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / CHECKPOINT_RELATIVE_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    """Write one stable terminal artifact and print its terminal state."""

    args = _parser().parse_args(argv)
    if args.output.is_file():
        existing = _load_json(args.output)
        current_sources = _source_hashes(REPO_ROOT)
        recorded = existing.get("source_artifact_hashes")
        stable = isinstance(recorded, Mapping) and all(
            recorded.get(path) == digest for path, digest in current_sources.items()
        )
        if (
            existing.get("execution_date") == args.date
            and stable
            and validate_artifact(existing) == []
        ):
            print(
                f"stable {args.output} belief_shadow_trace_ready_score={existing['belief_shadow_trace_ready_score']}"
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
