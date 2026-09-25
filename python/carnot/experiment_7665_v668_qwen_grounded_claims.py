"""Bounded paired Qwen extraction from native source and typed atom hints.

Only byte-scoped propositions are measured. Fixture truth is an exact oracle;
it says nothing about whole-answer accuracy or later science decisions.
Spec: REQ-REPORT-7665 and SCENARIO-REPORT-7665-*.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7658_v668_evidence_atoms as fixtures
from carnot.reporting import current_work_receipt as custody
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.verify.grounded_claims import check_pointers, parse_pointers
from carnot.verify.tool_source_atoms import extract_claims, parse_atoms, verify_answer


ROOT = Path(__file__).resolve().parents[2]
PILOT = Path("results/raw/experiment_7602_v664_evidence_requalification/pilot_model_inputs.jsonl")
RAW = Path("results/raw/experiment_7665_v668_qwen_grounded_claims")
RESULT = Path("results/experiment_7665_v668_qwen_grounded_claims.json")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
SEED = 7665
ARMS = ("source_only", "typed_index")
MODULE = "python/carnot/experiment_7665_v668_qwen_grounded_claims.py"
CAPABILITY = "python/carnot/verify/grounded_claims.py"
TEST = "tests/python/test_experiment_7665_v668_qwen_grounded_claims.py"
WRAPPER = "scripts/experiments/experiment_7665_v668_qwen_grounded_claims.py"
MANIFEST = {
    "requirement": "REQ-REPORT-7665",
    "tests": [TEST],
    "changed_modules": [MODULE, CAPABILITY],
    "static_paths": [WRAPPER],
}


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Keep a visible heartbeat while model or child work owns the process."""
    print(
        json.dumps(
            {
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **details,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def gate(
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Record the actual operand so resource absence can be diagnosed."""
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "eq",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def freeze_panel(pilots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Freeze source bytes and narrow truth before any model answer exists."""
    if len(pilots) != 8 or len({row["component_hash"] for row in pilots}) != 8:
        raise ValueError("eight_distinct_pilots_required")
    selected = [
        case
        for case in fixtures.fixture_cases()
        if case["id"].split("-")[-1] in {"00", "01", "08", "09"}
    ]
    if len(selected) != 16:
        raise ValueError("sixteen_fixture_sources_required")
    panel = []
    for record in pilots:
        panel.append(
            {
                "unit_id": record["component_hash"],
                "population": "pilot",
                "source": record["complete_source"],
                "answer": record["complete_answer"],
                "prior_exposure": True,
                "fixture_truth": None,
            }
        )
    for case in selected:
        panel.append(
            {
                "unit_id": case["id"],
                "population": "synthetic_control",
                "source": case["source"],
                "answer": case["answer"],
                "prior_exposure": True,
                "fixture_truth": case["truth"],
            }
        )
    for row in panel:
        row["truth"] = verify_answer(row["source"], row["answer"])
        row["claims"] = extract_claims(row["answer"])
        row["atoms"] = parse_atoms(row["source"])
        row["source_sha256"] = fixtures.digest(row["source"])
        row["answer_sha256"] = fixtures.digest(row["answer"])
    return panel


def make_request(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """Keep one schema, seed and original bytes across the paired prompts."""
    if arm not in ARMS:
        raise ValueError("invalid_arm")
    visible = {
        "complete_source": row["source"],
        "complete_answer": row["answer"],
        "claims": [
            {key: claim[key] for key in ("kind", "text", "byte_start", "byte_end")}
            for claim in row["claims"]
        ],
    }
    if arm == "typed_index":
        visible["typed_atom_index"] = [
            {
                key: atom[key]
                for key in (
                    "dialect",
                    "source_id",
                    "line",
                    "byte_start",
                    "byte_end",
                    "text",
                    "definitions",
                    "scopes",
                )
            }
            for atom in row["atoms"]
        ]
    instruction = (
        "/no_think\nReturn one JSON object with a pointers array. Each pointer has integer claim_index, "
        "integer source_byte_start, integer source_byte_end, and relation supports, contradicts, "
        "or unknown. Copy exact UTF-8 source byte offsets. Cite only visible structural claims. "
        "Do not certify causal or whole-answer meaning. Use an empty array when uncertain."
    )
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
        ],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def reduce_response(row: dict[str, Any], text: str, finish_reason: str) -> dict[str, Any]:
    """Keep malformed output and truncation in the denominator."""
    try:
        pointers = parse_pointers(text)
        valid = True
    except (ValueError, json.JSONDecodeError):
        pointers = []
        valid = False
    checked = check_pointers(row, pointers)
    return {
        "schema_valid": valid,
        "truncated": finish_reason == "length",
        "claim_count": len(row["claims"]),
        **checked,
    }


def blocked_artifact(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """External absence completes accounting without inventing science."""
    first = failures[0]
    return {
        "honest_verdict": f"complete_blocked_{first['check']}",
        "verdict_class": "blocked",
        "model_invoked": False,
        "MODEL_SPECS": [],
        "model_specs": [{"model": "none", "reason": "blocked_before_model_load"}],
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "rows": [],
        "proposition_rows": [],
        "gate_check_summary": {
            "failed_checks": [row["check"] for row in failures],
            "failed_count": len(failures),
            "first_failure": first,
        },
    }


def _preflight(
    root: Path, started: float
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:  # pragma: no cover
    """Authenticate inputs and free owned capacity before loading weights."""
    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot.inference.sota_models import cached_current_model

    named = [
        PILOT,
        Path("results/experiment_7658_v668_evidence_atoms.json"),
        Path("python/carnot/verify/tool_source_atoms.py"),
        Path("openspec/capabilities/research-reporting/spec.md"),
        Path("ops/exclusion_manifest.yaml"),
    ]
    checks = []
    hashes: dict[str, Any] = {
        "producers": {},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [str(RESULT)],
    }
    for relative in named:
        path = root / relative
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate(
                f"named_input:{relative.name}",
                "declared_input",
                str(path),
                "readable_nonempty_file",
                True,
                exists,
            )
        )
        if exists:
            hashes["producers"][str(relative)] = custody.sha256_file(path)
        else:
            hashes["missing_inputs"].append(str(relative))
    if not all(check["passed"] for check in checks):
        return checks, hashes, {}
    model = cached_current_model(preferred_quant="Q4_K_M")
    model_path = Path(str((model or {}).get("model_path") or root / "missing-model"))
    model_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and "Q4_K_M" in model_path.name
        and model_path.is_file()
        and model_path.stat().st_size > 15_000_000_000
    )
    checks.append(
        gate(
            "cached_q4_k_m_model",
            "local_model_cache",
            str(model_path),
            "hf_id_quantization_bytes",
            True,
            model_ok,
        )
    )
    if not model_ok:
        return checks, hashes, {}
    progress(started, "model_hash", "before", path=str(model_path))
    model_hash = custody.sha256_file(model_path)
    progress(started, "model_hash", "after", sha256=model_hash)
    hashes["producers"][str(model_path)] = model_hash
    registry = ownership.ProcessRegistry.current()
    inventory = ownership._current_inventory()
    selected, ownership_rows = ownership.select_owned_capacity(inventory, registry)
    checks.append(
        gate(
            "owned_cuda_capacity",
            "exp7630_cuda_ownership",
            str(root / RAW),
            "exclusive_idle_device_with_20000_mb",
            True,
            selected is not None,
        )
    )
    context = {
        "model": model,
        "model_path": model_path,
        "model_sha256": model_hash,
        "registry": registry,
        "selected": selected,
        "inventory": inventory,
        "ownership_rows": ownership_rows,
    }
    return checks, hashes, context


def _span(
    name: str, started: float, begin: float, planned: int, completed: int
) -> dict[str, Any]:  # pragma: no cover
    """Use disjoint monotonic stage bounds and explicit checkpoint position."""
    end = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": begin - started,
        "ended_offset_s": end - started,
        "duration_s": end - begin,
        "planned_units": planned,
        "completed_units": completed,
        "checkpoint_position": completed,
        "heartbeat_time_s": end - started,
    }


def _measure(
    root: Path, panel: list[dict[str, Any]], context: dict[str, Any], started: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:  # pragma: no cover
    """Own one server and checkpoint every bounded request before reduction."""
    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot import experiment_7604_v664_evidence_pilot as v664
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.experiment_7431_v651_arc_live_sentinel import _free_port
    from carnot.experiment_7581_v662_arc_bounded_canary import (
        _call_with_heartbeats,
        _observed_offload_layers,
        _owned_vram_mb,
        process_start_tick,
    )
    from carnot.gpu_lease_phase_journal import GpuLease, LeaseBusy, RecoveryError

    selected = context["selected"]
    assert selected is not None
    lease_start = time.monotonic()
    lease = None
    while lease is None:
        progress(started, "gpu_lease", "before", uuid=selected["uuid"])
        try:
            lease = GpuLease.acquire(
                runtime_dir=root / RAW / "gpu_leases",
                task_id="experiment_7665_v668_qwen_grounded_claims",
                device_uuid=str(selected["uuid"]),
                expected_model=str(context["model_path"]),
                vram_before_mb=int(selected["memory_used_mb"]),
                ttl_s=4800,
            )
        except (LeaseBusy, RecoveryError):
            if time.monotonic() - lease_start >= 180:
                raise TimeoutError("owned_cuda_lease_timeout") from None
            progress(started, "gpu_lease", "wait")
            time.sleep(15)
    progress(started, "gpu_lease", "after", lease_id=lease.lease_id)
    recheck = ownership.recheck_before_launch(
        str(selected["uuid"]),
        [ownership._current_inventory(), ownership._current_inventory()],
        context["registry"],
    )
    if recheck["passed"] is not True:
        lease.transition("terminal_blocked")
        lease.release()
        raise RuntimeError("foreign_or_capacity_recheck_failed")
    run_dir = root / RAW / "runs" / f"{int(time.time())}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    proposer = LocalGGUFProposer(
        repo_substr="Qwen3.8-27B",
        model_path=str(context["model_path"]),
        port=_free_port(),
        mtp=False,
        kv_quant="q8_0",
        use_chat_template=True,
        n_gpu_layers=999,
        n_ctx=32768,
        max_tokens=256,
        timeout=900,
        tries=1,
        extra_server_args=("-lv", "4"),
    )
    proposer.model_repository = MODEL_ID
    proposer.requested_model_path = str(context["model_path"])
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {
        "device_uuid": selected["uuid"],
        "lease_owner": lease.owner_receipt(),
        "model_load_attempted": 0,
        "model_load_completed": 0,
        "generation_attempted": 0,
    }
    try:
        lease.transition("admitted")
        lease.transition("loading")
        runtime["model_load_attempted"] = 1
        load_begin = time.monotonic()
        progress(started, "model_load", "before", uuid=selected["uuid"])
        healthy = _call_with_heartbeats(
            proposer._ensure_server, started=started, phase="exp7665_model_load"
        )
        progress(started, "model_load", "after", healthy=healthy)
        server_pid = getattr(getattr(proposer, "_proc", None), "pid", None)
        log_path = Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        owned_vram = _owned_vram_mb(server_pid)
        offload = v664._offload_receipt(log_path, owned_vram, _observed_offload_layers(log_path))
        runtime.update(
            {
                "model_load_s": time.monotonic() - load_begin,
                "server_pid": server_pid,
                "server_pid_start_ticks": process_start_tick(server_pid),
                "owned_vram_mb": owned_vram,
                "offload_layers": offload,
                "runtime_build": v664._runtime_build_receipt(),
                "server_props": proposer.server_props() if healthy else {},
            }
        )
        if not healthy or not offload.get("actual_offload") or not server_pid:
            raise RuntimeError("owned_q4_model_load_not_authenticated")
        runtime["model_load_completed"] = 1
        lease.transition("resident", vram_mb=int(owned_vram))
        lease.transition("inferencing")
        generation_start = time.monotonic()
        for unit_index, row in enumerate(panel, 1):
            for arm in ARMS:
                if time.monotonic() - generation_start >= 1800:
                    raise TimeoutError("generation_1800s_budget_exhausted")
                call_index = len(rows) + 1
                request = make_request(row, arm)
                request_path = run_dir / f"request_{call_index:02d}.json"
                response_path = run_dir / f"response_{call_index:02d}.json"
                custody.atomic_json(request_path, request)
                progress(
                    started,
                    "generation",
                    "before",
                    unit=unit_index,
                    completed=call_index - 1,
                    arm=arm,
                )
                call_begin = time.monotonic()
                runtime["generation_attempted"] += 1
                response_bytes = _call_with_heartbeats(
                    lambda: v664._post_json(
                        proposer._url() + "/v1/chat/completions",
                        request,
                        min(900.0, max(1.0, 1800 - (time.monotonic() - generation_start))),
                    ),
                    started=started,
                    phase=f"exp7665_generation_{call_index}",
                )
                response_path.write_bytes(response_bytes)
                response = json.loads(response_bytes)
                choice = response["choices"][0]
                message = choice["message"]
                content = str(message.get("content") or "")
                finish = str(choice.get("finish_reason") or "unknown")
                metrics = reduce_response(row, content, finish)
                usage = response.get("usage") or {}
                output_tokens = int(
                    usage.get("completion_tokens")
                    or v664._token_count(response, "predicted_n", "completion_tokens")
                    or 0
                )
                prompt_tokens = int(
                    usage.get("prompt_tokens")
                    or v664._token_count(response, "prompt_n", "prompt_tokens")
                    or 0
                )
                measured = {
                    "unit_id": row["unit_id"],
                    "population": row["population"],
                    "arm": arm,
                    "source_sha256": row["source_sha256"],
                    "answer_sha256": row["answer_sha256"],
                    "source": row["source"],
                    "answer": row["answer"],
                    "truth": row["truth"],
                    "fixture_truth": row["fixture_truth"],
                    "prior_exposure": row["prior_exposure"],
                    "request_path": str(request_path),
                    "request_sha256": custody.sha256_file(request_path),
                    "raw_response_path": str(response_path),
                    "raw_response_sha256": custody.sha256_file(response_path),
                    "response_text": content,
                    "finish_reason": finish,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "generation_s": time.monotonic() - call_begin,
                    "censored": metrics["truncated"],
                    "excluded": False,
                    "metrics": metrics,
                    "raw_metrics": {
                        key: value for key, value in metrics.items() if key != "proposition_rows"
                    },
                    "provenance": {
                        "model_id": MODEL_ID,
                        "server_pid": server_pid,
                        "gpu_uuid": selected["uuid"],
                    },
                }
                rows.append(measured)
                custody.atomic_json(run_dir / "checkpoint.json", {"rows": rows})
                progress(
                    started,
                    "generation",
                    "after",
                    completed=call_index,
                    output_tokens=output_tokens,
                    elapsed_call_s=measured["generation_s"],
                )
        runtime["generation_s"] = time.monotonic() - generation_start
        runtime["transport_authenticated"] = len(rows) == 48
    finally:
        progress(started, "model_unload", "before")
        proposer.stop()
        progress(started, "model_unload", "after")
        phase = str(lease.document.get("phase"))
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete" if len(rows) == 48 else "terminal_blocked")
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        runtime["lease_release"] = lease.release()
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
    return rows, runtime


def _acceptance(rows: list[dict[str, Any]], valid: bool) -> dict[str, Any]:
    """Separate protocol validity from unmeasured scientific benefit."""
    complete = len(rows) == 48
    definitions = {
        "validity": (
            valid,
            "Required current checks and exact readers govern validity.",
            {"rows": len(rows)},
        ),
        "readiness": (
            valid and complete and any(row["metrics"]["exact_supported"] for row in rows),
            "Bounded calls need at least one exact proposition to establish mechanism readiness.",
            {
                "completed_calls": len(rows),
                "planned_calls": 48,
                "exact_supported": sum(row["metrics"]["exact_supported"] for row in rows),
            },
        ),
        "coverage": (
            valid and complete and any(row["metrics"]["exact_supported"] for row in rows),
            "Independent source claims and unknowns stay in the denominator.",
            {
                "sources": len({row["unit_id"] for row in rows}),
                "exact_supported": sum(row["metrics"]["exact_supported"] for row in rows),
            },
        ),
        "probability_benefit": (
            None,
            "Proper loss needs independent labels and a held-back comparator.",
            {"confirmatory_groups": 0},
        ),
        "utility": (
            None,
            "A source pointer alone is not a typed action with an outcome.",
            {"decision_outcomes": 0},
        ),
        "retention": (None, "No delayed learning or restart was measured.", {"restarts": 0}),
        "freshness": (
            False,
            "Inherited pilots and exact fixtures are previously exposed.",
            {"fresh_confirmatory_groups": 0},
        ),
    }
    return {
        name: {"passed": passed, "principle": principle, "measured_operands": operands}
        for name, (passed, principle, operands) in definitions.items()
    }


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report precision, coverage and cost without pooling fixture and pilot units."""
    by_population: dict[str, Any] = {}
    for population in ("pilot", "synthetic_control"):
        by_arm = {}
        for arm in ARMS:
            selected = [
                row for row in rows if row["population"] == population and row["arm"] == arm
            ]
            exact = sum(row["metrics"]["exact_supported"] for row in selected)
            unsupported = sum(row["metrics"]["unsupported_certifications"] for row in selected)
            claims = sum(row["metrics"]["claim_count"] for row in selected)
            by_arm[arm] = {
                "independent_sources": len({row["unit_id"] for row in selected}),
                "exact_supported": exact,
                "unsupported_certifications": unsupported,
                "invalid_pointers": sum(row["metrics"]["invalid_pointers"] for row in selected),
                "claims": claims,
                "precision": exact / (exact + unsupported) if exact + unsupported else None,
                "coverage": exact / claims if claims else None,
                "generation_s": sum(row["generation_s"] for row in selected),
                "output_tokens": sum(row["output_tokens"] for row in selected),
            }
        by_population[population] = by_arm
    return by_population


def _artifact(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    spans: list[dict[str, Any]],
    runtime: dict[str, Any],
    started: float,
    date: str,
) -> dict[str, Any]:
    """Reduce per-arm evidence without promoting fixtures into science."""
    artifact = (
        blocked_artifact(failures)
        if failures
        else {
            "honest_verdict": "complete_null_bounded_grounded_claim_mechanism",
            "verdict_class": "null",
            "model_invoked": bool(runtime.get("model_load_attempted")),
            "MODEL_SPECS": [MODEL_ID] if runtime.get("model_load_attempted") else [],
            "model_specs": [
                {
                    "hf_id": MODEL_ID,
                    "quantization": "Q4_K_M",
                    "model_path": str(runtime.get("model_path", "")),
                }
            ],
            "inference_substrate": "owned_local_llama_cpp_paired_bounded_generation",
            "inference_substrate_class": "model_bounded_generation",
            "rows": rows,
            "proposition_rows": [
                {
                    "unit_id": row["unit_id"],
                    "population": row["population"],
                    "arm": row["arm"],
                    "arm_cost_s": row["generation_s"],
                    **claim,
                }
                for row in rows
                for claim in row["metrics"]["proposition_rows"]
            ],
            "gate_check_summary": {"failed_checks": [], "failed_count": 0},
        }
    )
    groups = {row["unit_id"] for row in rows}
    artifact.update(
        {
            "schema": "carnot.exp7665.v668.qwen_grounded_claims.v1",
            "experiment_id": "experiment_7665_v668_qwen_grounded_claims",
            "milestone": "2026.09.668",
            "run_date": date,
            "status": "complete",
            "planned_MODEL_SPECS": [MODEL_ID],
            "planned_inference_substrate_class": "model_bounded_generation",
            "execution_venue": "host",
            "execution_venue_details": {
                "host_pid": os.getpid(),
                "gpu_uuid": runtime.get("device_uuid"),
                "owned_server_pid": runtime.get("server_pid"),
            },
            "invocation_counts": {
                "model_loads_attempted": runtime.get("model_load_attempted", 0),
                "model_loads_completed": runtime.get("model_load_completed", 0),
                "generation_calls_attempted": runtime.get("generation_attempted", 0),
                "generation_calls_completed": len(rows),
                "forward_calls_attempted": runtime.get("generation_attempted", 0),
                "forward_calls_completed": len(rows),
                "input_tokens": sum(row["prompt_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "cancelled_calls": max(0, runtime.get("generation_attempted", 0) - len(rows)),
            },
            "phase_spans": spans,
            "duration_s": time.monotonic() - started,
            "random_seed": {
                "paired_generation": SEED,
                "fixture_selection": "00,01,08,09 per dialect",
            },
            "source_artifact_hashes": hashes,
            "preconditions_checked": checks,
            "sample_size_budget": {
                "independent_unit": "source_group",
                "intended": 24,
                "observed": len(groups),
                "eligible": len(groups),
                "excluded": 0,
                "censored": len({row["unit_id"] for row in rows if row["censored"]}),
                "pilot_groups": len(
                    {row["unit_id"] for row in rows if row["population"] == "pilot"}
                ),
                "synthetic_control_groups": len(
                    {row["unit_id"] for row in rows if row["population"] == "synthetic_control"}
                ),
                "prior_exposure": "Eight inherited pilots and sixteen exact oracle fixtures.",
                "claim_limits": "No whole-answer, probability, utility, retention, or fresh benefit.",
            },
            "paired_source_rows": [
                {
                    "unit_id": unit,
                    "population": next(row["population"] for row in rows if row["unit_id"] == unit),
                    "arms": {
                        row["arm"]: {
                            "exact_supported": row["metrics"]["exact_supported"],
                            "invalid_pointers": row["metrics"]["invalid_pointers"],
                            "unsupported_certifications": row["metrics"][
                                "unsupported_certifications"
                            ],
                            "generation_s": row["generation_s"],
                            "output_tokens": row["output_tokens"],
                        }
                        for row in rows
                        if row["unit_id"] == unit
                    },
                }
                for unit in sorted(groups)
            ],
            "population_arm_metrics": _summaries(rows),
            "gpu_lease_receipt": {
                key: runtime.get(key)
                for key in (
                    "device_uuid",
                    "lease_owner",
                    "server_pid",
                    "server_pid_start_ticks",
                    "owned_vram_mb",
                    "offload_layers",
                    "lease_release",
                )
            },
            "model_runtime_receipt": {
                key: runtime.get(key)
                for key in (
                    "model_load_s",
                    "generation_s",
                    "runtime_build",
                    "server_props",
                    "model_sha256",
                    "model_path",
                )
            },
            "model_revision": Path(str(runtime.get("model_path", ""))).parent.name
            if runtime.get("model_path")
            else None,
            "retirement_decision": {
                "scope": "V667/V668 source-pointer prompt mechanism only",
                "retire_if_same_verdict": True,
                "applied": bool(rows)
                and sum(row["metrics"]["exact_supported"] for row in rows) == 0,
                "principle": "Two bounded attempts with zero exact certified propositions close this prompt mechanism; external absence does not disprove it.",
            },
            "verifier_is_oracle": True,
            "whole_answer_benefit_claim": False,
            "flagged_adversarial": False,
            "validation_receipts": [],
            "terminal_reader_outcomes": {},
            "grounded_claim_measurement_complete_score": int(len(rows) == 48),
        }
    )
    artifact["acceptance_gate_results"] = _acceptance(rows, not failures)
    artifact["field_principles"] = {
        key: "Measured current work or explicitly bounded absence; "
        "fixture truth is an oracle and cannot establish science benefit."
        for key in artifact
    }
    artifact["reproducibility_checksum"] = custody.canonical_hash(
        {
            "input_hashes": hashes["producers"],
            "seed": SEED,
            "arms": ARMS,
            "max_tokens": 256,
            "reducer_sha256": custody.sha256_file(ROOT / CAPABILITY),
        }
    )
    return artifact


def independent_reduce(path: Path) -> dict[str, Any]:  # pragma: no cover
    """Read raw bytes in a new process and reject changed pointers or labels."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact["rows"]
    if artifact["verdict_class"] == "blocked":
        return {"passed": not rows and not artifact["model_invoked"], "groups": 0}
    pilots = [json.loads(line) for line in (ROOT / PILOT).read_text(encoding="utf-8").splitlines()]
    panel = {row["unit_id"]: row for row in freeze_panel(pilots)}
    if len(rows) != 48 or len(panel) != 24:
        return {"passed": False, "reason": "incomplete_panel"}
    arms: dict[str, set[str]] = {}
    for row in rows:
        original = panel.get(row["unit_id"])
        if original is None or row["population"] != original["population"]:
            return {"passed": False, "reason": "source_roster_drift"}
        if row["source"] != original["source"] or row["answer"] != original["answer"]:
            return {"passed": False, "reason": "source_bytes_drift"}
        request_path = Path(row["request_path"])
        response_path = Path(row["raw_response_path"])
        if (
            not request_path.is_file()
            or not response_path.is_file()
            or custody.sha256_file(request_path) != row["request_sha256"]
            or custody.sha256_file(response_path) != row["raw_response_sha256"]
        ):
            return {"passed": False, "reason": "raw_byte_drift"}
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request != make_request(original, row["arm"]):
            return {"passed": False, "reason": "request_or_label_drift"}
        response = json.loads(response_path.read_text(encoding="utf-8"))
        choice = response["choices"][0]
        text = str(choice["message"].get("content") or "")
        finish = str(choice.get("finish_reason") or "unknown")
        if reduce_response(original, text, finish) != row["metrics"]:
            return {"passed": False, "reason": "pointer_reduction_drift"}
        arms.setdefault(row["unit_id"], set()).add(row["arm"])
    fixture = next(row for row in panel.values() if row["population"] == "synthetic_control")
    control = check_pointers(
        {**fixture, "source": ""},
        [{"claim_index": 0, "source_byte_start": 0, "source_byte_end": 1, "relation": "supports"}],
    )
    paired = len(arms) == 24 and all(value == set(ARMS) for value in arms.values())
    return {
        "passed": paired and control["exact_supported"] == 0,
        "groups": len(arms),
        "calls": len(rows),
        "erasure_control_exact_supported": control["exact_supported"],
    }


def cold_replay(path: Path) -> dict[str, Any]:  # pragma: no cover
    """Replay one candidate without the loaded server or producer memory."""
    reduced = independent_reduce(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = custody.canonical_hash(
        {
            "input_hashes": value["source_artifact_hashes"]["producers"],
            "seed": SEED,
            "arms": ARMS,
            "max_tokens": 256,
            "reducer_sha256": custody.sha256_file(ROOT / CAPABILITY),
        }
    )
    return {
        "passed": reduced["passed"] and expected == value["reproducibility_checksum"],
        "reduction": reduced,
    }


def _validation_commands(
    root: Path, private: Path
) -> list[validation.CommandSpec]:  # pragma: no cover
    """Freeze the affected file set and use private serial pytest paths."""
    basetemp = private / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    return validation.build_scoped_commands(
        root,
        MANIFEST["tests"],
        MANIFEST["changed_modules"],
        static_paths=MANIFEST["static_paths"],
        basetemp=basetemp,
        coverage_file=private / ".coverage.exp7665",
    )


def _terminal_commands(
    root: Path, candidate: Path
) -> list[validation.CommandSpec]:  # pragma: no cover
    """Run fresh readers against one immutable candidate path."""
    python = str(root / ".venv/bin/python")
    cli = str(root / WRAPPER)
    return [
        validation.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", cli, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
        validation.CommandSpec(
            "independent_reduction",
            (python, "-u", cli, "--independent-reduce", str(candidate)),
            "persisted_raw_rows",
            180,
        ),
        validation.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
        validation.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
    ]


def _finalize(
    artifact: dict[str, Any],
    receipts: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
    started: float,
) -> None:  # pragma: no cover
    """Invalid receipts disqualify instead of masquerading as a null."""
    artifact["validation_receipts"] = receipts + terminal
    artifact["terminal_reader_outcomes"] = {
        row["name"]: {
            "passed": row["passed"],
            "exit_code": row["exit_code"],
            "log_sha256": row["log_sha256"],
        }
        for row in terminal
    }
    artifact["flagged_adversarial"] = not next(
        row["passed"] for row in terminal if row["name"] == "adversarial_verify"
    )
    valid = all(row["passed"] for row in receipts + terminal)
    if not valid:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["grounded_claim_measurement_complete_score"] = 0
    artifact["acceptance_gate_results"] = _acceptance(artifact["rows"], valid)
    if not valid:
        for gate_result in artifact["acceptance_gate_results"].values():
            gate_result["passed"] = False
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = {
        key: "Measured current work or bounded absence; "
        "fixtures are exact oracles and cannot prove whole-answer benefit."
        for key in artifact
    }


def run_experiment(root: Path, date: str, output: Path) -> int:  # pragma: no cover
    """Authenticate, measure, validate and publish the terminal candidate."""
    started = time.monotonic()
    progress(started, "startup", "before", root=str(root.resolve()))
    root = root.resolve()
    if date != "20260925":
        raise SystemExit("--date must be 20260925")
    destination = output if output.is_absolute() else root / output
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7665-"))
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, context = _preflight(root, started)
    failures = [row for row in checks if not row["passed"]]
    spans.append(_span("preconditions", started, begin, len(checks), len(checks) - len(failures)))
    progress(started, "preconditions", "after", failed=len(failures))
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    if not failures:
        begin = time.monotonic()
        progress(started, "freeze_panel", "before")
        pilots = [
            json.loads(line) for line in (root / PILOT).read_text(encoding="utf-8").splitlines()
        ]
        panel = freeze_panel(pilots)
        (root / RAW).mkdir(parents=True, exist_ok=True)
        custody.atomic_json(root / RAW / "frozen_panel.json", panel)
        custody.atomic_json(root / RAW / "frozen_affected_validation_manifest.json", MANIFEST)
        hashes["pre_gate_receipts"]["frozen_panel"] = custody.sha256_file(
            root / RAW / "frozen_panel.json"
        )
        spans.append(_span("freeze_panel", started, begin, 24, len(panel)))
        progress(started, "freeze_panel", "after", groups=len(panel))
        begin = time.monotonic()
        progress(started, "measurement", "before", calls=48)
        try:
            rows, runtime = _measure(root, panel, context, started)
            runtime["model_path"] = str(context["model_path"])
            runtime["model_sha256"] = context["model_sha256"]
        except BaseException as error:
            failure = gate(
                "owned_model_transport",
                "exp7630_owned_q4_server",
                str(root / RAW),
                "forty_eight_bounded_calls",
                True,
                f"{type(error).__name__}:{error}",
            )
            checks.append(failure)
            failures.append(failure)
        spans.append(_span("measurement", started, begin, 48, len(rows)))
        progress(started, "measurement", "after", completed=len(rows), failed=bool(failures))
    artifact = _artifact(rows, failures, checks, hashes, spans, runtime, started, date)
    progress(started, "scoped_validation", "before")
    begin = time.monotonic()
    receipts = validation.run_commands(
        root,
        _validation_commands(root, private),
        log_dir=private / "logs" / "scoped",
        heartbeat_s=45,
    )
    spans.append(_span("scoped_validation", started, begin, len(receipts), len(receipts)))
    progress(started, "scoped_validation", "after", passed=all(row["passed"] for row in receipts))
    if not all(row["passed"] for row in receipts):
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["acceptance_gate_results"] = _acceptance(rows, False)
    artifact["validation_receipts"] = receipts
    candidate = private / "terminal_candidate.json"
    custody.atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "before", candidate=str(candidate))
    begin = time.monotonic()
    terminal = validation.run_commands(
        root,
        _terminal_commands(root, candidate),
        log_dir=private / "logs" / "terminal",
        heartbeat_s=45,
    )
    spans.append(_span("terminal_readers", started, begin, len(terminal), len(terminal)))
    progress(started, "terminal_readers", "after", passed=all(row["passed"] for row in terminal))
    _finalize(artifact, receipts, terminal, started)
    (root / RAW).mkdir(parents=True, exist_ok=True)
    custody.atomic_json(
        root / RAW / "terminal_reader_outcomes.json", artifact["terminal_reader_outcomes"]
    )
    custody.atomic_json(destination, artifact)
    progress(
        started,
        "publish",
        "after",
        path=str(destination),
        sha256=custody.sha256_file(destination),
        verdict=artifact["honest_verdict"],
    )
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Dispatch the live run or a read-only fresh-process check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = cold_replay(args.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    if args.independent_reduce:
        result = independent_reduce(args.independent_reduce)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    return run_experiment(ROOT, args.date, args.output)
