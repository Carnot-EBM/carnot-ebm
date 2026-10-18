"""Qwen evidence-view diagnostic. REQ-REPORT-7759."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

from carnot import experiment_7745_v674_qwen_localization as prior
from carnot.reporting import current_work_receipt as custody
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.verify.source_alignment import sentence_spans

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7759_v675_qwen_evidence_views")
RESULT = Path("results/experiment_7759_v675_qwen_evidence_views.json")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
ARMS = ("canonical", "paired", "source_withheld")
SEED = 7759
MAX_PROMPT_BYTES = 24_000
SCOPE = {
    "tests": ["tests/python/test_experiment_7759_v675_qwen_evidence_views.py"],
    "changed_modules": ["python/carnot/experiment_7759_v675_qwen_evidence_views.py"],
    "static_paths": ["scripts/experiments/experiment_7759_v675_qwen_evidence_views.py"],
    "specs": ["REQ-REPORT-7759"],
    "e2e": [
        "task_owned_gguf_transport",
        "fresh_process_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "strict_row_consistency",
    ],
}
PRINCIPLE = "The exposed family, raw bytes, and validation bound every claim."


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush an elapsed-time boundary with a completed-unit count."""
    tail = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[exp7759] {phase} {event} elapsed_s={time.monotonic() - started:.2f} {tail}", flush=True
    )


def prepare_basetemp(path: Path) -> None:
    """Create every nested pytest parent before a child process starts."""
    path.mkdir(parents=True, exist_ok=True)


def indexed_windows(source: str, arm: str) -> list[dict[str, Any]]:
    """Keep exact UTF-8 sentence bytes and add adjacent pairs or triples."""
    if arm == "source_withheld":
        return []
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    parts = sentence_spans(source.encode("utf-8"))
    width = 3 if arm == "canonical" else 2
    windows = list(parts) + [
        b"".join(parts[i : i + width]) for i in range(max(0, len(parts) - width + 1))
    ]
    return [{"index": i, "text": data.decode("utf-8")} for i, data in enumerate(windows)]


def make_request(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """Use one decision instruction; only indexed source windows vary."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    visible: dict[str, Any] = {"original_answer": row["answer"]}
    if arm != "source_withheld":
        visible["complete_source"] = row["source"]
    visible["source_windows"] = indexed_windows(row["source"], arm)
    instruction = (
        "/no_think\nDecide whether the original answer contains unsupported content "
        "from the available source evidence. Return exactly one JSON object "
        "with decision (unsupported, supported, or abstain), "
        "probability_unsupported (a number from 0 to 1), and quote "
        "(an exact source substring or empty string). An index is a locator, "
        "not proof. Abstain when uncertain."
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


def parse_reply(row: dict[str, Any], arm: str, text: str, finish: str) -> dict[str, Any]:
    """Score strict syntax and stated risk without inventing a withheld label."""
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = None
    valid = (
        isinstance(parsed, dict)
        and set(parsed) == {"decision", "probability_unsupported", "quote"}
        and parsed.get("decision") in {"unsupported", "supported", "abstain"}
        and type(parsed.get("probability_unsupported")) in {float, int}
        and math.isfinite(parsed["probability_unsupported"])
        and 0 <= parsed["probability_unsupported"] <= 1
        and isinstance(parsed.get("quote"), str)
    )
    usable = bool(valid and finish == "stop")
    probability = float(parsed["probability_unsupported"]) if usable else None
    label = (
        bool(row["annotation_types"])
        if arm != "source_withheld" and row.get("annotation_types") is not None
        else None
    )
    quote = parsed["quote"] if valid else None
    return {
        "parse_valid": bool(valid),
        "typed_decision": parsed["decision"] if usable else None,
        "probability_unsupported": probability,
        "quote": quote,
        "quote_valid": bool(quote and arm != "source_withheld" and row["source"].count(quote) == 1),
        "human_binary_unsupported": label,
        "brier": (probability - float(label)) ** 2
        if probability is not None and label is not None
        else None,
        "truncated": finish == "length",
        "censored": finish in {"length", "timeout", "error"},
        "missing": finish == "missing",
        "semantic_verified": False,
    }


def reduce_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce all dispositions while counting each independent family once."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family_id"], {})[row["arm"]] = row
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        sample = [row for row in rows if row["arm"] == arm]
        valid = [row for row in sample if row["metrics"]["probability_unsupported"] is not None]
        briers = [row["metrics"]["brier"] for row in sample if row["metrics"]["brier"] is not None]
        by_arm[arm] = {
            "denominator": len(sample),
            "parse_valid": sum(row["metrics"]["parse_valid"] for row in sample),
            "valid_probability": len(valid),
            "brier_count": len(briers),
            "brier": sum(briers) / len(briers) if briers else None,
            "typed_decisions": {
                decision: sum(row["metrics"]["typed_decision"] == decision for row in sample)
                for decision in ("unsupported", "supported", "abstain")
            },
            "censored": sum(row["metrics"]["censored"] for row in sample),
            "unstarted": sum(row["disposition"].startswith("unstarted") for row in sample),
            "input_tokens": sum(row["input_tokens"] for row in sample),
            "output_tokens": sum(row["output_tokens"] for row in sample),
        }
    pairs = [
        (
            arms["canonical"]["metrics"]["probability_unsupported"],
            arms["paired"]["metrics"]["probability_unsupported"],
        )
        for arms in grouped.values()
        if "canonical" in arms and "paired" in arms
    ]
    distances = [abs(a - b) for a, b in pairs if a is not None and b is not None]
    return {
        "effective_independent_n": len(grouped),
        "complete_three_arm_families": sum(set(arms) == set(ARMS) for arms in grouped.values()),
        "paired_probability_count": len(distances),
        "paired_probability_disagreement": sum(distances) / len(distances) if distances else None,
        "by_arm": by_arm,
    }


def capture_family(
    row: dict[str, Any],
    transport: Callable[[dict[str, Any]], dict[str, Any] | bytes],
    raw_dir: Path,
    started: float,
    *,
    max_prompt_bytes: int = MAX_PROMPT_BYTES,
    deadline: float = float("inf"),
    output_left: int = 18432,
) -> list[dict[str, Any]]:
    """Save exact requests and replies and give every planned arm a row."""
    output: list[dict[str, Any]] = []
    for arm in ARMS:
        request = make_request(row, arm)
        size = len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
        reason = (
            "unstarted_prompt_over_budget"
            if size > max_prompt_bytes
            else "unstarted_time_budget"
            if time.monotonic() >= deadline
            else "unstarted_output_budget"
            if output_left <= 0
            else None
        )
        receipt: dict[str, Any] = {
            "family_id": row["family_id"],
            "arm": arm,
            "official_split": row["official_split"],
            "prior_exposure": True,
            "denominator": 1,
            "excluded": False,
            "source_sha256": hashlib.sha256(row["source"].encode()).hexdigest(),
            "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest(),
            "prompt_bytes": size,
            "raw_request_path": None,
            "raw_request_sha256": None,
            "raw_response_path": None,
            "raw_response_sha256": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": 0.0,
            "finish_reason": "missing",
            "response_text": "",
            "disposition": reason or "started",
        }
        if reason is None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{row['family_id']}_{arm}"
            request_path = raw_dir / f"{stem}_request.json"
            response_path = raw_dir / f"{stem}_response.json"
            custody.atomic_json(request_path, request)
            receipt["raw_request_path"] = str(request_path)
            receipt["raw_request_sha256"] = custody.sha256_file(request_path)
            progress(
                started,
                "generation",
                "before",
                family=row["family_id"],
                arm=arm,
                completed_units=len(output),
            )
            begin = time.monotonic()
            try:
                answer = transport(request)
                response_bytes = (
                    answer
                    if isinstance(answer, bytes)
                    else json.dumps(answer, sort_keys=True).encode()
                )
                response_path.write_bytes(response_bytes)
                response = json.loads(response_bytes)
                choice = response["choices"][0]
                receipt["response_text"] = str(choice["message"].get("content") or "")
                receipt["finish_reason"] = str(choice.get("finish_reason") or "missing")
                usage = response.get("usage") or {}
                receipt["input_tokens"] = int(usage.get("prompt_tokens") or 0)
                receipt["output_tokens"] = int(usage.get("completion_tokens") or 0)
                receipt["disposition"] = "completed"
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                response_path.write_text(
                    json.dumps({"transport_error": f"{type(error).__name__}:{error}"})
                )
                receipt["finish_reason"] = "error"
                receipt["disposition"] = "censored_transport_error"
            receipt["elapsed_s"] = time.monotonic() - begin
            receipt["raw_response_path"] = str(response_path)
            receipt["raw_response_sha256"] = custody.sha256_file(response_path)
            output_left -= receipt["output_tokens"]
            progress(
                started,
                "generation",
                "after",
                family=row["family_id"],
                arm=arm,
                completed_units=len(output) + 1,
                output_tokens=receipt["output_tokens"],
            )
        receipt["metrics"] = parse_reply(
            row, arm, receipt["response_text"], receipt["finish_reason"]
        )
        receipt["censored"] = receipt["metrics"]["censored"]
        output.append(receipt)
    return output


def gate(
    check: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Record an exact prerequisite operand and the bytes that supplied it."""
    return {
        "check": check,
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": custody.sha256_file(path) if path.is_file() else None,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(
    root: Path, started: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate Exp7745 separately from its frozen pre-gate panel."""
    old_path = root / prior.RESULT
    panel_path = root / prior.RAW / "frozen_panel.json"
    hashes: dict[str, Any] = {"valid_producers": {}, "pre_gate_receipts": {}, "missing_custody": []}
    checks = [
        gate(
            "exp7745_result_exists",
            "exp7745",
            old_path,
            "readable_nonempty_bytes",
            True,
            old_path.is_file() and old_path.stat().st_size > 0,
        ),
        gate(
            "exp7745_panel_exists",
            "exp7745_pre_gate",
            panel_path,
            "readable_nonempty_bytes",
            True,
            panel_path.is_file() and panel_path.stat().st_size > 0,
        ),
    ]
    if any(not check["passed"] for check in checks):
        hashes["missing_custody"] = [
            check["artifact_path"] for check in checks if not check["passed"]
        ]
        return [], checks, hashes, {}
    old = json.loads(old_path.read_text())
    expected = {
        "milestone": "2026.09.674",
        "qwen_localization_complete_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
    }
    checks.extend(
        gate("exp7745_qualified", "exp7745", old_path, field, value, old.get(field))
        for field, value in expected.items()
    )
    checks.append(
        gate(
            "exp7745_families",
            "exp7745",
            old_path,
            "paired_family_results.paired_families",
            24,
            (old.get("paired_family_results") or {}).get("paired_families"),
        )
    )
    old_panel_hash = (
        (old.get("source_artifact_hashes") or {})
        .get("pre_gate_receipts", {})
        .get(str(prior.RAW / "frozen_panel.json"))
    )
    checks.append(
        gate(
            "exp7745_panel_hash",
            "exp7745_pre_gate",
            panel_path,
            "sha256",
            old_panel_hash,
            custody.sha256_file(panel_path),
        )
    )
    hashes["valid_producers"][str(prior.RESULT)] = custody.sha256_file(old_path)
    hashes["pre_gate_receipts"][str(prior.RAW / "frozen_panel.json")] = custody.sha256_file(
        panel_path
    )
    if any(not check["passed"] for check in checks):
        return [], checks, hashes, {}
    panel, inherited, inherited_hashes, context = prior.prepare_panel(root, started)
    checks.extend(inherited)
    hashes["exp7745_reauthenticated_sources"] = inherited_hashes
    frozen = json.loads(panel_path.read_text())
    checks.append(
        gate(
            "same_exp7745_panel",
            "exp7745_pre_gate",
            panel_path,
            "reconstructed_panel_equals_frozen",
            True,
            panel == frozen,
        )
    )
    checks.append(
        gate(
            "same_qwen_bytes",
            "exp7745",
            old_path,
            "current_model_receipts.model_sha256",
            context.get("model_sha256"),
            (old.get("current_model_receipts") or {}).get("model_sha256"),
        )
    )
    if any(not check["passed"] for check in checks):
        return [], checks, hashes, context
    return frozen, checks, hashes, context


def owned_capture(
    root: Path, panel: list[dict[str, Any]], context: dict[str, Any], started: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use the qualified owned CUDA server, preserving active offload receipts."""
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
    from carnot.gpu_lease_phase_journal import GpuLease

    selected = context["selected"]
    lease = GpuLease.acquire(
        runtime_dir=root / RAW / "gpu_leases",
        task_id="experiment_7759_v675_qwen_evidence_views",
        device_uuid=str(selected["uuid"]),
        expected_model=str(context["model_path"]),
        vram_before_mb=int(selected["memory_used_mb"]),
        ttl_s=4800,
    )
    runtime: dict[str, Any] = {
        "device_uuid": selected["uuid"],
        "model_path": str(context["model_path"]),
        "model_sha256": context["model_sha256"],
        "quantization": "Q4_K_M",
        "lease_owner": lease.owner_receipt(),
        "model_load_attempted": 0,
        "model_load_completed": 0,
        "generation_attempted": 0,
        "active_window_memory_mb": [],
    }
    rows: list[dict[str, Any]] = []
    proposer = None
    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        recheck = ownership.recheck_before_launch(
            str(selected["uuid"]),
            [ownership._current_inventory(), ownership._current_inventory()],
            context["registry"],
        )
        if recheck["passed"] is not True:
            raise RuntimeError("foreign_or_capacity_recheck_failed")
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
        lease.transition("admitted")
        lease.transition("loading")
        runtime["model_load_attempted"] = 1
        progress(started, "model_load", "before", completed_units=0, uuid=selected["uuid"])
        load_begin = time.monotonic()
        healthy = _call_with_heartbeats(
            proposer._ensure_server, started=started, phase="exp7759_model_load"
        )
        runtime["model_load_s"] = time.monotonic() - load_begin
        progress(
            started,
            "model_load",
            "after",
            completed_units=int(bool(healthy)),
            duration_s=runtime["model_load_s"],
        )
        if not healthy or not getattr(proposer, "_proc", None):
            raise RuntimeError("owned_qwen_load_failed")
        runtime["model_load_completed"] = 1
        runtime["server_pid"] = proposer._proc.pid
        runtime["server_pid_start_ticks"] = process_start_tick(proposer._proc.pid)
        runtime["server_props"] = proposer.server_props()
        log_path = Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        owned_vram = _owned_vram_mb(proposer._proc.pid)
        runtime["owned_vram_mb"] = owned_vram
        runtime["offload_layers"] = v664._offload_receipt(
            log_path, owned_vram, _observed_offload_layers(log_path)
        )
        runtime["runtime_build"] = v664._runtime_build_receipt()
        if not runtime["offload_layers"].get("actual_offload"):
            raise RuntimeError("qwen_gpu_offload_not_authenticated")
        template = str(runtime["server_props"].get("chat_template") or "")
        if hashlib.sha256(template.encode()).hexdigest() != context["chat_template_sha256"]:
            raise RuntimeError("chat_template_changed_after_load")
        lease.transition("resident", vram_mb=int(owned_vram))
        lease.transition("inferencing")
        run_dir = root / RAW / "runs" / f"{int(time.time())}-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        runtime["run_dir"] = str(run_dir)
        deadline = time.monotonic() + 3300

        def transport(request: dict[str, Any]) -> bytes:
            runtime["generation_attempted"] += 1
            return _call_with_heartbeats(
                lambda: v664._post_json(proposer._url() + "/v1/chat/completions", request, 900),
                started=started,
                phase="exp7759_generation",
            )

        output_left = 18432
        for index, family in enumerate(panel, 1):
            family_rows = capture_family(
                family, transport, run_dir, started, deadline=deadline, output_left=output_left
            )
            rows.extend(family_rows)
            output_left -= sum(row["output_tokens"] for row in family_rows)
            runtime["active_window_memory_mb"].append(
                {
                    "completed_families": index,
                    "server_pid": proposer._proc.pid,
                    "owned_vram_mb": _owned_vram_mb(proposer._proc.pid),
                }
            )
            custody.atomic_json(
                run_dir / "checkpoint.json", {"rows": rows, "completed_families": index}
            )
            progress(
                started,
                "measurement",
                "checkpoint",
                completed_units=index,
                calls=runtime["generation_attempted"],
                output_left=output_left,
            )
    finally:
        progress(started, "model_unload", "before", completed_units=len(rows))
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after", completed_units=len(rows))
        phase = str(lease.document.get("phase"))
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete" if len(rows) == 72 else "terminal_blocked")
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        runtime["lease_release"] = lease.release()
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
    return rows, runtime


def phase_span(
    name: str, started: float, begin: float, units: int, checkpoint: Path
) -> dict[str, Any]:
    """Keep a measured monotonic span and exact checkpoint identity."""
    end = time.monotonic()
    return {
        "phase": name,
        "start_s": begin - started,
        "end_s": end - started,
        "duration_s": end - begin,
        "completed_units": units,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": custody.sha256_file(checkpoint) if checkpoint.is_file() else None,
    }


def build_artifact(
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    runtime: dict[str, Any],
    spans: list[dict[str, Any]],
    duration: float,
    date: str,
) -> dict[str, Any]:
    """Separate transport completion, scientific null, and failed validation."""
    reduced = reduce_rows(rows)
    failed = [check for check in checks if not check["passed"]]
    blocked = bool(failed and not runtime.get("model_load_attempted"))
    calls = [row for row in rows if row["raw_request_path"]]
    complete = len(rows) == 72 and reduced["complete_three_arm_families"] == 24
    verdict_class = "blocked" if blocked else "null" if complete and not failed else "partial"
    verdict = (
        "complete_blocked_" + failed[0]["check"]
        if blocked
        else "complete_null_exposed_evidence_view_pilot"
        if verdict_class == "null"
        else "complete_partial_owned_capture"
    )
    gates = {
        name: {"passed": None, "measured_operands": {}, "principle": PRINCIPLE}
        for name in (
            "validity",
            "readiness",
            "probability_quality",
            "decision_benefit",
            "retention",
            "efficiency",
        )
    }
    gates["validity"].update(passed=not failed, measured_operands={"failed_checks": len(failed)})
    gates["readiness"].update(passed=False, measured_operands={"terminal_rows": len(rows)})
    gates["probability_quality"]["measured_operands"] = {
        arm: reduced["by_arm"][arm]["brier"] for arm in ("canonical", "paired")
    }
    gates["decision_benefit"]["measured_operands"] = {
        "paired_probability_disagreement": reduced["paired_probability_disagreement"]
    }
    gates["retention"].update(passed=complete, measured_operands={"terminal_rows": len(rows)})
    gates["efficiency"]["measured_operands"] = {
        "calls": len(calls),
        "input_tokens": sum(row["input_tokens"] for row in calls),
        "output_tokens": sum(row["output_tokens"] for row in calls),
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7759.v675.qwen_evidence_views.v1",
        "experiment_id": "exp7759-qwen-evidence-views",
        "milestone": "2026.09.675",
        "run_date": date,
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failed,
        "acceptance_gate_results": gates,
        "rows": rows,
        "paired_family_results": reduced,
        "sample_size_budget": {
            "intended_families": 24,
            "eligible_families": 24 if rows else 0,
            "started_families": len({row["family_id"] for row in calls}),
            "completed_families": sum(
                all(arm in arms for arm in ARMS)
                for arms in (
                    {row["arm"] for row in rows if row["family_id"] == family}
                    for family in {row["family_id"] for row in rows}
                )
            ),
            "excluded_families": 0,
            "censored_families": len({row["family_id"] for row in rows if row["censored"]}),
            "effective_independent_n": reduced["effective_independent_n"],
            "intended_calls": 72,
            "started_calls": len(calls),
            "completed_calls": sum(row["disposition"] == "completed" for row in calls),
            "unstarted_calls": sum(row["disposition"].startswith("unstarted") for row in rows),
            "output_token_ceiling": 18432,
            "output_tokens": sum(row["output_tokens"] for row in calls),
            "arms_are_repeated_views": True,
        },
        "claim_scope": "exposed_development_diagnostic_only; no learned_head_gate",
        "fresh_generalization_eligible": False,
        "verifier_is_oracle": False,
        "inference_substrate": "live_llm_inference"
        if runtime.get("model_load_completed")
        else "no_model_load",
        "inference_substrate_class": "model_bounded_generation" if calls else "no_model_load",
        "planned_inference_substrate": "live_llm_inference",
        "planned_inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": [MODEL_ID] if runtime.get("model_load_attempted") else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_specs": [
            {
                "name": "Qwen3.8-27B",
                "hf_id": MODEL_ID,
                "model_path": runtime.get("model_path"),
                "model_sha256": runtime.get("model_sha256"),
                "quantization": "Q4_K_M",
            }
        ]
        if runtime.get("model_load_attempted")
        else [],
        "model_invocation_counts": {
            "loads": runtime.get("model_load_attempted", 0),
            "loads_completed": runtime.get("model_load_completed", 0),
            "calls": runtime.get("generation_attempted", 0),
            "input_tokens": sum(row["input_tokens"] for row in calls),
            "output_tokens": sum(row["output_tokens"] for row in calls),
        },
        "gpu_provenance": {
            key: runtime.get(key)
            for key in (
                "device_uuid",
                "model_path",
                "model_sha256",
                "quantization",
                "owned_vram_mb",
                "offload_layers",
                "active_window_memory_mb",
                "runtime_build",
                "model_load_s",
            )
        },
        "token_budget_rows": [
            {
                key: row[key]
                for key in (
                    "family_id",
                    "arm",
                    "input_tokens",
                    "output_tokens",
                    "finish_reason",
                    "elapsed_s",
                    "disposition",
                )
            }
            for row in rows
        ],
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {
            "generation": SEED,
            "arm_order": "fixed canonical, paired, source_withheld",
        },
        "reproducibility_checksum": custody.canonical_hash(
            {
                "sources": hashes,
                "seed": SEED,
                "arms": ARMS,
                "prompt_bytes_ceiling": MAX_PROMPT_BYTES,
                "max_tokens": 256,
                "code_sha256": custody.sha256_file(Path(__file__)),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": [],
            "terminal_readers": [],
            "unrelated_full_suite_debt": [],
        },
        "pilot_evidence_ready_score": 0,
        "current_model_receipts": runtime,
    }
    artifact["field_principles"] = {
        **{key: PRINCIPLE for key in artifact},
        **{f"acceptance_gate_{key}": PRINCIPLE for key in gates},
    }
    return artifact


def cold_reduce(path: Path) -> dict[str, Any]:
    """Reopen raw bytes and recompute every started row in a fresh reader."""
    artifact = json.loads(path.read_text())
    rows = artifact["rows"]
    if artifact["verdict_class"] == "blocked":
        return {"passed": not rows and not artifact["MODEL_SPECS"], "calls": 0}
    panel_path = ROOT / prior.RAW / "frozen_panel.json"
    panel = {row["family_id"]: row for row in json.loads(panel_path.read_text())}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = row["family_id"], row["arm"]
        if key in seen or key[0] not in panel or key[1] not in ARMS:
            return {"passed": False, "reason": "row_identity"}
        seen.add(key)
        original = panel[key[0]]
        if (
            row["source_sha256"] != hashlib.sha256(original["source"].encode()).hexdigest()
            or row["answer_sha256"] != hashlib.sha256(original["answer"].encode()).hexdigest()
            or row["prompt_bytes"]
            != len(json.dumps(make_request(original, key[1]), ensure_ascii=False).encode())
        ):
            return {"passed": False, "reason": "original_bytes_or_prompt"}
        if row["raw_request_path"] is None:
            if not row["disposition"].startswith("unstarted"):
                return {"passed": False, "reason": "missing_started_request"}
            continue
        request_path, response_path = Path(row["raw_request_path"]), Path(row["raw_response_path"])
        if (
            custody.sha256_file(request_path) != row["raw_request_sha256"]
            or custody.sha256_file(response_path) != row["raw_response_sha256"]
            or json.loads(request_path.read_text()) != make_request(original, key[1])
        ):
            return {"passed": False, "reason": "raw_hash_or_request"}
        response = json.loads(response_path.read_bytes())
        if "transport_error" in response:
            text, finish, input_tokens, output_tokens = "", "error", 0, 0
        else:
            choice = response["choices"][0]
            text, finish = (
                str(choice["message"].get("content") or ""),
                str(choice.get("finish_reason") or "missing"),
            )
            usage = response.get("usage") or {}
            input_tokens, output_tokens = (
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
        if (
            text != row["response_text"]
            or finish != row["finish_reason"]
            or input_tokens != row["input_tokens"]
            or output_tokens != row["output_tokens"]
            or output_tokens > 256
            or row["metrics"] != parse_reply(original, key[1], text, finish)
        ):
            return {"passed": False, "reason": "raw_metric_or_token"}
    if len(seen) != len(panel) * 3 or reduce_rows(rows) != artifact["paired_family_results"]:
        return {"passed": False, "reason": "aggregate_or_roster"}
    return {
        "passed": True,
        "families": len(panel),
        "calls": sum(row["raw_request_path"] is not None for row in rows),
    }


def terminal_commands(root: Path, candidate: Path) -> list[validation.CommandSpec]:
    """Register each independent reader against one exact candidate path."""
    python = str(root / ".venv/bin/python")
    entry = str(root / SCOPE["static_paths"][0])
    return [
        validation.CommandSpec(
            "independent_cold_replay",
            (python, "-u", entry, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
        validation.CommandSpec(
            "independent_reduction",
            (python, "-u", entry, "--reduce-candidate", str(candidate)),
            "exact_terminal_candidate",
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


def run_experiment(root: Path, date: str, output: Path) -> int:
    """Prequalify, capture once, validate exact rows, then atomically publish."""
    started = time.monotonic()
    progress(started, "startup", "before", completed_units=0)
    root = root.resolve(strict=True)
    if date != "20260927":
        raise ValueError("date_must_be_20260927")
    destination = output if output.is_absolute() else root / output
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    frozen_scope = raw_dir / "frozen_affected_scope.json"
    if json.loads(frozen_scope.read_text()) != SCOPE:
        raise ValueError("affected_scope_changed_after_freeze")
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7759-"))
    prepare_basetemp(private / "pytest")
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    progress(started, "preconditions", "before", completed_units=0)
    try:
        panel, checks, hashes, context = preflight(root, started)
    except (OSError, ValueError, KeyError, TypeError) as error:
        panel, context = [], {}
        checks = [
            gate(
                "source_reconstruction",
                "exp7745",
                root / prior.RAW,
                "authenticated_original_panel",
                True,
                f"{type(error).__name__}:{error}",
            )
        ]
        hashes = {
            "valid_producers": {},
            "pre_gate_receipts": {},
            "missing_custody": [str(root / prior.RAW)],
        }
    panel_path = raw_dir / "frozen_panel.json"
    if panel and all(check["passed"] for check in checks):
        custody.atomic_json(panel_path, panel)
        hashes["pre_gate_receipts"][str(RAW / "frozen_panel.json")] = custody.sha256_file(
            panel_path
        )
    spans.append(phase_span("preconditions", started, begin, len(checks), panel_path))
    failed = [check for check in checks if not check["passed"]]
    progress(started, "preconditions", "after", completed_units=len(checks), failures=len(failed))

    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    if not failed:
        begin = time.monotonic()
        progress(started, "measurement", "before", completed_units=0, families=len(panel))
        try:
            rows, runtime = owned_capture(root, panel, context, started)
        except BaseException as error:
            runs = sorted((raw_dir / "runs").glob(f"*-{os.getpid()}"))
            checkpoint = runs[-1] / "checkpoint.json" if runs else None
            if checkpoint and checkpoint.is_file():
                rows = json.loads(checkpoint.read_text())["rows"]
            runtime = {
                "model_load_attempted": 1,
                "generation_attempted": sum(row["raw_request_path"] is not None for row in rows),
                "model_path": str(context["model_path"]),
                "model_sha256": context["model_sha256"],
                "error": f"{type(error).__name__}:{error}",
            }
            checks.append(
                gate(
                    "owned_capture",
                    "exp7630_cuda_ownership",
                    raw_dir,
                    "terminal_rows",
                    72,
                    len(rows),
                )
            )
        spans.append(phase_span("measurement", started, begin, len(rows), raw_dir / "runs"))
        progress(started, "measurement", "after", completed_units=len(rows))
    rows_path = raw_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )
    hashes["pre_gate_receipts"][str(RAW / "rows.jsonl")] = custody.sha256_file(rows_path)
    artifact = build_artifact(
        rows, checks, hashes, runtime, spans, time.monotonic() - started, date
    )

    begin = time.monotonic()
    progress(started, "scoped_validation", "before", completed_units=0)
    commands = validation.build_scoped_commands(
        root,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7759",
    )
    receipts = validation.run_commands(
        root,
        commands,
        log_dir=raw_dir / "validation" / "affected",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=45,
    )
    spans.append(phase_span("scoped_validation", started, begin, len(receipts), frozen_scope))
    artifact["validation_receipts"]["required_commands"] = receipts
    valid_checks = validation.reduce_required_checks(receipts)["required_checks_passed"]
    progress(
        started, "scoped_validation", "after", completed_units=len(receipts), passed=valid_checks
    )
    if not valid_checks:
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    candidate = raw_dir / "terminal_candidate.json"
    custody.atomic_json(candidate, artifact)
    begin = time.monotonic()
    progress(started, "terminal_readers", "before", completed_units=0)
    terminal = validation.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=raw_dir / "validation" / "terminal",
        extra_env={"JAX_PLATFORMS": "cpu"},
        heartbeat_s=45,
    )
    spans.append(phase_span("terminal_readers", started, begin, len(terminal), candidate))
    progress(
        started,
        "terminal_readers",
        "after",
        completed_units=len(terminal),
        passed=all(item["passed"] for item in terminal),
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["validation_receipts"]["terminal_candidate_sha256"] = custody.sha256_file(candidate)
    adversarial = next(item for item in terminal if item["name"] == "adversarial_verify")
    try:
        report = json.loads((root / adversarial["log_path"]).read_text())
        flagged = bool(report["flagged_count"])
    except (OSError, ValueError, KeyError, TypeError):
        flagged = True
    artifact["flagged_adversarial"] = flagged
    all_valid = valid_checks and all(item["passed"] for item in terminal) and not flagged
    if not all_valid:
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
    complete = (
        len(rows) == 72 and artifact["paired_family_results"]["complete_three_arm_families"] == 24
    )
    actual_qwen = bool(
        runtime.get("model_load_completed")
        and (runtime.get("offload_layers") or {}).get("actual_offload")
    )
    artifact["pilot_evidence_ready_score"] = int(
        complete and actual_qwen and all_valid and not failed
    )
    artifact["acceptance_gate_results"]["readiness"].update(
        passed=bool(artifact["pilot_evidence_ready_score"]),
        measured_operands={
            "terminal_rows": len(rows),
            "actual_qwen_cuda": actual_qwen,
            "required_checks_passed": all_valid,
        },
    )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = {
        **{key: PRINCIPLE for key in artifact},
        **{f"acceptance_gate_{key}": PRINCIPLE for key in artifact["acceptance_gate_results"]},
    }
    custody.atomic_json(destination, artifact)
    progress(
        started,
        "publication",
        "after",
        completed_units=len(rows),
        path=destination,
        verdict=artifact["honest_verdict"],
        sha256=custody.sha256_file(destination),
    )
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: list[str] | None = None) -> int:
    """Accept the fixed run date and read-only independent replay modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--reduce-candidate", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        receipt = cold_reduce(args.cold_replay)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 0 if receipt["passed"] else 1
    if args.reduce_candidate:
        artifact = json.loads(args.reduce_candidate.read_text())
        reduced = reduce_rows(artifact["rows"])
        passed = reduced == artifact["paired_family_results"]
        print(json.dumps({"passed": passed, "reduced": reduced}, sort_keys=True), flush=True)
        return 0 if passed else 1
    return run_experiment(ROOT, args.date, args.output)
