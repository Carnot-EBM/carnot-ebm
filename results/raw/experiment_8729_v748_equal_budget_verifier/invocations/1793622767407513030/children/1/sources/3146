"""Run the exposed, paired Qwen draft pilot. REQ-REPORT-7729."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, Callable

from carnot import experiment_7716_v672_qwen_semantic_pilot as prior
from carnot.reporting import current_work_receipt as custody
from carnot.reporting import experiment_7303_validation_scope as validation


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7729_v673_qwen_draft_pilot")
RESULT = Path("results/experiment_7729_v673_qwen_draft_pilot.json")
MODEL_ID = prior.MODEL_ID
ARMS = ("direct_schema", "draft_schema", "draft_plain")
SEED = 7729
SCOPE = {
    "tests": ["tests/python/test_experiment_7729_v673_qwen_draft_pilot.py"],
    "changed_modules": ["python/carnot/experiment_7729_v673_qwen_draft_pilot.py"],
    "static_paths": ["scripts/experiments/experiment_7729_v673_qwen_draft_pilot.py"],
    "specs": ["REQ-REPORT-7729", "REQ-VERIFY-7729"],
    "e2e": ["task_owned_capture_each_arm", "task_cold_raw_replay"],
}
PRINCIPLE = "Measured evidence bounds the claim and downstream use."
DECISIONS = frozenset({"support", "contradiction", "insufficient_evidence"})


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print each real boundary and completed count while work is running."""
    tail = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[exp7729] {phase} {event} elapsed_s={time.monotonic() - started:.2f} {tail}", flush=True
    )


def gate(
    check: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep exact operands for every absent external prerequisite."""
    return prior.gate(check, upstream, path, field, expected, observed)


def arm_order(family_id: str) -> tuple[str, ...]:
    """Choose a stable independent permutation within each source family."""
    arms = list(ARMS)
    seed = int(hashlib.sha256(f"{SEED}:{family_id}".encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(arms)
    return tuple(arms)


def make_request(
    row: dict[str, Any], arm: str, stage: str, draft: str | None = None
) -> dict[str, Any]:
    """Keep original source and answer visible at every bounded call."""
    if (
        arm not in ARMS
        or stage not in {"draft", "answer"}
        or (arm == "direct_schema" and stage == "draft")
    ):
        raise ValueError("unplanned_arm_or_stage")
    if stage == "answer" and arm != "direct_schema" and draft is None:
        raise ValueError("draft_required")
    source = json.dumps(
        {"complete_source": row["source"], "original_answer": row["answer"]}, ensure_ascii=False
    )
    if stage == "draft":
        instruction = "/no_think\nWrite a short private draft that checks the original answer against the complete source. Keep exact possible quotations. Do not emit the final answer."
    else:
        instruction = (
            "/no_think\nJudge the complete original answer from the source as support, contradiction, "
            "or insufficient_evidence. Give one exact original-source quotation. "
            "If evidence is insufficient, say insufficient_evidence. "
            + (
                "Return exactly one JSON object with decision and quote."
                if arm != "draft_plain"
                else "Answer freely, but state decision and quote explicitly."
            )
        )
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": source}]
    if draft is not None:
        messages.append({"role": "assistant", "content": draft})
        messages.append({"role": "user", "content": "Now give the final decision and quotation."})
    request: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 256 if arm == "direct_schema" else 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if stage == "answer" and arm != "draft_plain":
        request["response_format"] = {"type": "json_object"}
    return request


def score_response(
    source: str, text: str, finish: str, annotation_types: list[str] | None
) -> dict[str, Any]:
    """Score strict JSON, loose plain text, and binary human truth separately."""
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        pass
    syntax = (
        isinstance(parsed, dict)
        and set(parsed) == {"decision", "quote"}
        and parsed.get("decision") in DECISIONS
        and isinstance(parsed.get("quote"), str)
    )
    decision = parsed["decision"] if syntax else None
    quote = parsed["quote"] if syntax else None
    if decision is None:
        match = re.search(
            r"(?im)^decision:\s*(support|contradiction|insufficient_evidence)\s*$", text
        )
        found = re.search(r'(?im)^quote:\s*"?(.+?)"?\s*$', text)
        if match and found:
            decision, quote = match.group(1), found.group(1).strip('"')
    human_unsupported = bool(annotation_types) if annotation_types is not None else None
    binary = decision != "support" if decision is not None else None
    return {
        "syntax_valid": bool(syntax),
        "decision": decision,
        "quote": quote,
        "quote_valid": bool(quote and source.count(quote) == 1),
        "missing_answer": decision is None,
        "truncated": finish == "length",
        "censored": finish in {"length", "timeout", "error"},
        "unknown": decision == "insufficient_evidence",
        "human_binary_unsupported": human_unsupported,
        "binary_accuracy": bool(binary == human_unsupported)
        if binary is not None and human_unsupported is not None and finish == "stop"
        else False,
        "three_way_accuracy": None,
        "semantic_verified": False,
    }


def execute_family(
    row: dict[str, Any],
    transport: Callable[[dict[str, Any]], dict[str, Any] | bytes],
    raw_dir: Path,
    started: float,
) -> list[dict[str, Any]]:
    """Send five real calls, keep each byte receipt, and checkpoint one family."""
    family_rows = []
    for arm in arm_order(row["family_id"]):
        calls = []
        draft = None
        stages = ("answer",) if arm == "direct_schema" else ("draft", "answer")
        for stage in stages:
            request = make_request(row, arm, stage, draft)
            stem = f"{row['family_id']}_{arm}_{stage}"
            request_path, response_path = (
                raw_dir / f"{stem}_request.json",
                raw_dir / f"{stem}_response.json",
            )
            custody.atomic_json(request_path, request)
            progress(
                started,
                "generation",
                "before",
                family=row["family_id"],
                arm=arm,
                stage=stage,
                completed_units=len(calls),
            )
            call_start = time.monotonic()
            answer = transport(request)
            response_bytes = (
                answer if isinstance(answer, bytes) else json.dumps(answer, sort_keys=True).encode()
            )
            response_path.write_bytes(response_bytes)
            response = json.loads(response_bytes)
            choice = response["choices"][0]
            content = str(choice["message"].get("content") or "")
            finish = str(choice.get("finish_reason") or "unknown")
            usage = response.get("usage") or {}
            receipt = {
                "stage": stage,
                "request_path": str(request_path),
                "request_sha256": custody.sha256_file(request_path),
                "raw_response_path": str(response_path),
                "raw_response_sha256": custody.sha256_file(response_path),
                "response_text": content,
                "finish_reason": finish,
                "input_tokens": int(usage.get("prompt_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or 0),
                "latency_s": time.monotonic() - call_start,
            }
            calls.append(receipt)
            if stage == "draft":
                draft = content
            progress(
                started,
                "generation",
                "after",
                family=row["family_id"],
                arm=arm,
                stage=stage,
                completed_units=len(calls),
                output_tokens=receipt["output_tokens"],
            )
        final = calls[-1]
        metrics = score_response(
            row["source"], final["response_text"], final["finish_reason"], row["annotation_types"]
        )
        family_rows.append(
            {
                "family_id": row["family_id"],
                "arm": arm,
                "official_split": row["official_split"],
                "prior_exposure": True,
                "source_sha256": hashlib.sha256(row["source"].encode()).hexdigest(),
                "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest(),
                "calls": calls,
                "metrics": metrics,
                "raw_metrics": metrics,
                "input_tokens": sum(call["input_tokens"] for call in calls),
                "output_tokens": sum(call["output_tokens"] for call in calls),
                "latency_s": sum(call["latency_s"] for call in calls),
                "denominator": 1,
                "excluded": False,
                "exclusion_reason": None,
                "censored": metrics["censored"]
                or any(call["finish_reason"] in {"timeout", "error"} for call in calls),
            }
        )
    custody.atomic_json(raw_dir / f"checkpoint_{row['family_id']}.json", family_rows)
    progress(started, "family", "after", family=row["family_id"], completed_units=len(family_rows))
    return family_rows


def reduce_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Treat families as independent units and include malformed outputs."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family_id"], {})[row["arm"]] = row
    complete = {family: arms for family, arms in grouped.items() if set(arms) == set(ARMS)}
    by_arm = {}
    for arm in ARMS:
        sample = [arms[arm] for arms in complete.values()]
        by_arm[arm] = {
            "denominator": len(sample),
            "binary_accuracy_numerator": sum(r["metrics"]["binary_accuracy"] for r in sample),
            "syntax_valid_numerator": sum(r["metrics"]["syntax_valid"] for r in sample),
            "quote_valid_numerator": sum(r["metrics"]["quote_valid"] for r in sample),
            "unknown_numerator": sum(r["metrics"]["unknown"] for r in sample),
            "missing_answer_numerator": sum(r["metrics"]["missing_answer"] for r in sample),
            "truncated_numerator": sum(r["metrics"]["truncated"] for r in sample),
            "input_tokens": sum(r["input_tokens"] for r in sample),
            "output_tokens": sum(r["output_tokens"] for r in sample),
            "latency_s": sum(r["latency_s"] for r in sample),
        }
    comparisons = {}
    for control in ("direct_schema", "draft_plain"):
        deltas = [
            int(arms["draft_schema"]["metrics"]["binary_accuracy"])
            - int(arms[control]["metrics"]["binary_accuracy"])
            for arms in complete.values()
        ]
        n = len(deltas)
        draws = []
        if n:
            rng = random.Random(SEED)
            draws = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(2048))
        comparisons[f"draft_schema_vs_{control}"] = {
            "paired_n": n,
            "binary_accuracy_delta": sum(deltas) / n if n else None,
            "paired_bootstrap_95_interval": [draws[51], draws[1996]] if n else None,
            "per_family_deltas": dict(zip(complete, deltas)),
        }
    return {"paired_families": len(complete), "by_arm": by_arm, "comparisons": comparisons}


def preflight(
    root: Path, started: float
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate upstream bytes, runtime, chat template and owned capacity."""
    checks, hashes, context = prior.preflight(root, started)
    old_path = root / prior.RESULT
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    prior_runtime = old.get("current_model_receipts") or {}
    template = (prior_runtime.get("server_props") or {}).get("chat_template")
    checks.extend(
        [
            gate("prior_runtime_receipt", "exp7716", old_path, "readable_json", True, bool(old)),
            gate(
                "same_model_bytes",
                "exp7716",
                old_path,
                "model_sha256",
                context.get("model_sha256"),
                prior_runtime.get("model_sha256"),
            ),
            gate(
                "qwen_chat_template",
                "exp7716",
                old_path,
                "nonempty_chat_template",
                True,
                bool(template),
            ),
        ]
    )
    hashes["flagged_historical_evidence"][str(prior.RESULT)] = (
        custody.sha256_file(old_path) if old_path.is_file() else None
    )
    context["chat_template_sha256"] = hashlib.sha256(str(template or "").encode()).hexdigest()
    return checks, hashes, context


def owned_capture(
    root: Path, panel: list[dict[str, Any]], context: dict[str, Any], started: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse the owned Qwen server, cooperative lease and 55-second heartbeat."""
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
        task_id="experiment_7729_v673_qwen_draft_pilot",
        device_uuid=str(selected["uuid"]),
        expected_model=str(context["model_path"]),
        vram_before_mb=int(selected["memory_used_mb"]),
        ttl_s=4800,
    )
    runtime: dict[str, Any] = {
        "device_uuid": selected["uuid"],
        "model_path": str(context["model_path"]),
        "model_sha256": context["model_sha256"],
        "chat_template_sha256": context["chat_template_sha256"],
        "lease_owner": lease.owner_receipt(),
        "model_load_attempted": 0,
        "model_load_completed": 0,
        "generation_attempted": 0,
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
        progress(started, "model_load", "before", uuid=selected["uuid"])
        load_start = time.monotonic()
        healthy = _call_with_heartbeats(
            proposer._ensure_server, started=started, phase="exp7729_model_load"
        )
        runtime["model_load_s"] = time.monotonic() - load_start
        progress(
            started, "model_load", "after", healthy=healthy, duration_s=runtime["model_load_s"]
        )
        if not healthy or not getattr(proposer, "_proc", None):
            raise RuntimeError("owned_qwen_load_failed")
        runtime["model_load_completed"] = 1
        runtime["server_pid"] = proposer._proc.pid
        runtime["server_props"] = proposer.server_props()
        log_path = Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        owned_vram = _owned_vram_mb(proposer._proc.pid)
        runtime["server_pid_start_ticks"] = process_start_tick(proposer._proc.pid)
        runtime["owned_vram_mb"] = owned_vram
        runtime["offload_layers"] = v664._offload_receipt(
            log_path, owned_vram, _observed_offload_layers(log_path)
        )
        runtime["runtime_build"] = v664._runtime_build_receipt()
        if not runtime["offload_layers"].get("actual_offload"):
            raise RuntimeError("qwen_gpu_offload_not_authenticated")
        if (
            hashlib.sha256(
                str(runtime["server_props"].get("chat_template") or "").encode()
            ).hexdigest()
            != context["chat_template_sha256"]
        ):
            raise RuntimeError("chat_template_changed_after_load")
        lease.transition("resident", vram_mb=int(owned_vram))
        lease.transition("inferencing")
        run_dir = root / RAW / "runs" / f"{int(time.time())}-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=False)

        def transport(request: dict[str, Any]) -> bytes:
            runtime["generation_attempted"] += 1
            return _call_with_heartbeats(
                lambda: v664._post_json(proposer._url() + "/v1/chat/completions", request, 900),
                started=started,
                phase="exp7729_generation",
            )

        for index, family in enumerate(panel, 1):
            rows.extend(execute_family(family, transport, run_dir, started))
            custody.atomic_json(
                run_dir / "checkpoint.json", {"rows": rows, "completed_families": index}
            )
            progress(
                started,
                "measurement",
                "checkpoint",
                completed_units=index,
                calls=runtime["generation_attempted"],
            )
        runtime["run_dir"] = str(run_dir)
    finally:
        progress(started, "model_unload", "before")
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after")
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


def cold_reduce(path: Path, raw_dir: Path | None = None) -> dict[str, Any]:
    """Reopen frozen source and exact request/response bytes in a fresh reader."""
    artifact = json.loads(path.read_text())
    rows = artifact["rows"]
    if artifact["verdict_class"] == "blocked":
        return {"passed": not rows and not artifact["model_invoked"], "families": 0, "calls": 0}
    panel_path = (raw_dir or ROOT / RAW) / "frozen_panel.json"
    panel = {row["family_id"]: row for row in json.loads(panel_path.read_text())}
    seen = set()
    call_count = 0
    for row in rows:
        key = (row["family_id"], row["arm"])
        if key in seen or key[0] not in panel or key[1] not in ARMS:
            return {"passed": False, "reason": "family_arm_identity"}
        seen.add(key)
        original = panel[key[0]]
        calls = row["calls"]
        if len(calls) != (1 if key[1] == "direct_schema" else 2):
            return {"passed": False, "reason": "call_count"}
        draft = None
        for call in calls:
            request_path = Path(call["request_path"])
            response_path = Path(call["raw_response_path"])
            if (
                custody.sha256_file(request_path) != call["request_sha256"]
                or custody.sha256_file(response_path) != call["raw_response_sha256"]
            ):
                return {"passed": False, "reason": "raw_hash"}
            if json.loads(request_path.read_text()) != make_request(
                original, key[1], call["stage"], draft
            ):
                return {"passed": False, "reason": "request_mismatch"}
            response = json.loads(response_path.read_text())
            choice = response["choices"][0]
            content = str(choice["message"].get("content") or "")
            if (
                content != call["response_text"]
                or str(choice.get("finish_reason") or "unknown") != call["finish_reason"]
            ):
                return {"passed": False, "reason": "response_mismatch"}
            usage = response.get("usage") or {}
            if (
                int(usage.get("completion_tokens") or 0) != call["output_tokens"]
                or int(usage.get("prompt_tokens") or 0) != call["input_tokens"]
            ):
                return {"passed": False, "reason": "token_mismatch"}
            if call["output_tokens"] > (256 if key[1] == "direct_schema" else 128):
                return {"passed": False, "reason": "token_budget"}
            draft = content if call["stage"] == "draft" else draft
            call_count += 1
        final = calls[-1]
        metrics = score_response(
            original["source"],
            final["response_text"],
            final["finish_reason"],
            original["annotation_types"],
        )
        if metrics != row["metrics"] or row["output_tokens"] != sum(
            call["output_tokens"] for call in calls
        ):
            return {"passed": False, "reason": "metric_mismatch"}
    paired = reduce_pairs(rows)
    return {
        "passed": len(seen) == len(panel) * 3 and paired == artifact["paired_family_results"],
        "families": paired["paired_families"],
        "calls": call_count,
    }


def build_artifact(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    hashes: dict[str, Any],
    runtime: dict[str, Any],
    spans: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    """Bound the pilot claim by actual paired calls and current checks."""
    paired = reduce_pairs(rows)
    complete = len(rows) == 72 and paired["paired_families"] == 24
    invoked = bool(runtime.get("model_load_attempted"))
    blocked = bool(failures and not invoked)
    verdict_class = "blocked" if blocked else "null" if complete and not failures else "partial"
    verdict = (
        "complete_blocked_" + failures[0]["check"]
        if blocked
        else "complete_null_exposed_draft_pilot"
        if verdict_class == "null"
        else "complete_partial_owned_capture"
    )
    gate_names = (
        "validity",
        "readiness",
        "brier_score",
        "decision_cost",
        "coverage",
        "retention",
        "efficiency",
    )
    gates = {
        name: {"passed": None, "measured_operands": {}, "principle": PRINCIPLE}
        for name in gate_names
    }
    gates["validity"].update(
        passed=not failures, measured_operands={"failed_checks": len(failures)}
    )
    gates["readiness"].update(
        passed=complete and not failures,
        measured_operands={"paired_families": paired["paired_families"]},
    )
    gates["coverage"].update(
        passed=complete,
        measured_operands={
            "observed_families": len({r["family_id"] for r in rows}),
            "intended_families": 24,
        },
    )
    gates["efficiency"]["measured_operands"] = {
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "duration_s": duration,
    }
    fields = (
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "claim_scope",
        "inference_substrate",
        "inference_substrate_class",
        "MODEL_SPECS",
        "model_invoked",
        "execution_venue",
        "phase_spans",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "preconditions_checked",
        "validation_receipts",
        "verifier_is_oracle",
        "qwen_pilot_complete_score",
        "paired_family_results",
        "current_model_receipts",
        "semantic_target_limits",
    )
    calls = [call for row in rows for call in row["calls"]]
    return {
        "schema": "carnot.exp7729.v673.qwen_draft_pilot.v1",
        "experiment_id": "exp7729-qwen-draft-pilot",
        "milestone": "2026.09.673",
        "run_date": "20260926",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "paired_family_results": paired,
        "sample_size_budget": {
            "intended_families": 24,
            "observed_families": len({r["family_id"] for r in rows}),
            "eligible_families": 24 if rows else 0,
            "excluded_families": 0,
            "censored_families": len({r["family_id"] for r in rows if r["censored"]}),
            "effective_independent_families": paired["paired_families"],
            "intended_calls": 120,
            "observed_calls": len(calls),
            "output_token_ceiling": 18432,
            "arms_and_calls_are_not_independent_families": True,
        },
        "claim_scope": "development_only",
        "fresh_generalization_eligible": False,
        "inference_substrate": "owned_local_llama_cpp_bounded_generation"
        if invoked
        else "no_model_load",
        "inference_substrate_class": "model_bounded_generation" if invoked else "no_model_load",
        "planned_inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": [MODEL_ID] if invoked else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_specs": [{"hf_id": MODEL_ID, "model_path": runtime.get("model_path")}]
        if invoked
        else [],
        "model_invoked": invoked,
        "invocation_counts": {
            "loads": runtime.get("model_load_attempted", 0),
            "forwards": runtime.get("generation_attempted", 0),
            "generations": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "failures": runtime.get("generation_failures", 0),
            "cancellations": runtime.get("cancellations", 0),
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host_pid": os.getpid(),
            "server_pid": runtime.get("server_pid"),
            "gpu_uuid": runtime.get("device_uuid") if invoked else None,
        },
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {
            "generation": SEED,
            "arm_order": "sha256:7729:family_id",
            "bootstrap": SEED,
        },
        "reproducibility_checksum": custody.canonical_hash(
            {
                "hashes": hashes,
                "seed": SEED,
                "arms": ARMS,
                "reducer": custody.sha256_file(Path(__file__)),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": [],
        "validation_receipts": {
            "frozen_affected_scope": SCOPE,
            "required_commands": [],
            "terminal_readers": [],
            "unrelated_full_suite_debt": [],
        },
        "verifier_is_oracle": False,
        "qwen_pilot_complete_score": 0,
        "current_model_receipts": runtime,
        "semantic_target_limits": "Human binary unsupported labels do not establish contradiction versus insufficient-evidence accuracy; exact quotes certify location only.",
        "field_principles": {
            **{field: PRINCIPLE for field in fields},
            **{f"acceptance_gate_{name}": PRINCIPLE for name in gate_names},
        },
        "activation": False,
        "production_promotion": False,
    }


def span(name: str, started: float, begin: float, units: int, checkpoint: Path) -> dict[str, Any]:
    """Record one disjoint monotonic phase and its actual checkpoint bytes."""
    end = time.monotonic()
    return {
        "phase": name,
        "start_s": begin - started,
        "end_s": end - started,
        "duration_s": end - begin,
        "run_date": "20260926",
        "heartbeat_times_s": [end - started],
        "completed_units": units,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": custody.sha256_file(checkpoint) if checkpoint.is_file() else None,
    }


def terminal_commands(root: Path, candidate: Path) -> list[validation.CommandSpec]:
    """Send the exact candidate to an independent replay and strict readers."""
    python = str(root / ".venv/bin/python")
    return [
        validation.CommandSpec(
            "independent_cold_replay",
            (python, "-u", str(root / SCOPE["static_paths"][0]), "--cold-replay", str(candidate)),
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
    """Freeze, validate, capture once, cold-read, and publish atomically."""
    started = time.monotonic()
    progress(started, "startup", "before", root=root.resolve())
    root = root.resolve(strict=True)
    if date != "20260926":
        raise ValueError("date_must_be_20260926")
    destination = output if output.is_absolute() else root / output
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7729-"))
    (private / "pytest").mkdir(parents=True, exist_ok=True)
    scope_path = raw_dir / "frozen_affected_scope.json"
    custody.atomic_json(scope_path, SCOPE)
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, context = preflight(root, started)
    failures = [item for item in checks if not item["passed"]]
    spans.append(span("preconditions", started, begin, len(checks), scope_path))
    progress(started, "preconditions", "after", completed_units=len(checks), failures=len(failures))
    panel: list[dict[str, Any]] = []
    panel_path = raw_dir / "frozen_panel.json"
    if not failures:
        begin = time.monotonic()
        progress(started, "freeze_panel", "before")
        try:
            panel, roster = prior.freeze_panel(root, started)
            checks.append(
                gate(
                    "exposed_panel_size",
                    "RAGTruth",
                    panel_path,
                    "distinct_families",
                    24,
                    len({r["family_id"] for r in panel}),
                )
            )
            checks.append(
                gate(
                    "fresh_roster_disjoint",
                    "exp7715",
                    panel_path,
                    "fresh_roster_intersection",
                    0,
                    roster["fresh_roster_count"],
                )
            )
            if all(item["passed"] for item in checks):
                custody.atomic_json(panel_path, panel)
                hashes["pre_gate_receipts"][str(RAW / "frozen_panel.json")] = custody.sha256_file(
                    panel_path
                )
        except (OSError, ValueError, KeyError) as error:
            checks.append(
                gate(
                    "panel_authentication",
                    "RAGTruth",
                    panel_path,
                    "complete_exposed_panel",
                    True,
                    f"{type(error).__name__}:{error}",
                )
            )
        failures = [item for item in checks if not item["passed"]]
        spans.append(span("freeze_panel", started, begin, len(panel), panel_path))
        progress(
            started, "freeze_panel", "after", completed_units=len(panel), failures=len(failures)
        )
    begin = time.monotonic()
    progress(started, "scoped_validation", "before")
    commands = validation.build_scoped_commands(
        root,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7729",
    )
    receipts = validation.run_commands(
        root, commands, log_dir=raw_dir / "validation" / "affected", heartbeat_s=45
    )
    spans.append(
        span(
            "scoped_validation", started, begin, len(receipts), raw_dir / "validation" / "affected"
        )
    )
    progress(
        started,
        "scoped_validation",
        "after",
        completed_units=len(receipts),
        passed=all(r["passed"] for r in receipts),
    )
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    if not failures and all(r["passed"] for r in receipts):
        begin = time.monotonic()
        progress(started, "measurement", "before", families=len(panel), max_calls=120)
        try:
            rows, runtime = owned_capture(root, panel, context, started)
        except BaseException as error:
            runs = sorted((raw_dir / "runs").glob(f"*-{os.getpid()}"))
            checkpoint = runs[-1] / "checkpoint.json" if runs else None
            if checkpoint and checkpoint.is_file():
                rows = json.loads(checkpoint.read_text())["rows"]
            runtime = {
                "model_load_attempted": 1,
                "generation_attempted": sum(len(r["calls"]) for r in rows),
                "model_path": str(context["model_path"]),
                "model_sha256": context["model_sha256"],
                "device_uuid": context["selected"]["uuid"],
                "error": f"{type(error).__name__}:{error}",
            }
            checks.append(
                gate(
                    "owned_capture",
                    "exp7630_owned_qwen_server",
                    raw_dir,
                    "paired_families",
                    24,
                    len({r["family_id"] for r in rows}),
                )
            )
            failures = [item for item in checks if not item["passed"]]
        spans.append(span("measurement", started, begin, len(rows), raw_dir / "runs"))
        progress(started, "measurement", "after", completed_units=len(rows), failures=len(failures))
    rows_path = raw_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    hashes["pre_gate_receipts"][str(RAW / "rows.jsonl")] = custody.sha256_file(rows_path)
    artifact = build_artifact(rows, failures, hashes, runtime, spans, time.monotonic() - started)
    artifact["preconditions_checked"] = checks
    artifact["validation_receipts"]["required_commands"] = receipts
    if not all(r["passed"] for r in receipts):
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    candidate = raw_dir / "terminal_candidate.json"
    custody.atomic_json(candidate, artifact)
    begin = time.monotonic()
    progress(started, "terminal_readers", "before")
    terminal = validation.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=raw_dir / "validation" / "terminal",
        heartbeat_s=45,
    )
    spans.append(span("terminal_readers", started, begin, len(terminal), candidate))
    progress(
        started,
        "terminal_readers",
        "after",
        completed_units=len(terminal),
        passed=all(r["passed"] for r in terminal),
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["flagged_adversarial"] = not next(
        r["passed"] for r in terminal if r["name"] == "adversarial_verify"
    )
    if not all(r["passed"] for r in receipts + terminal):
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
        for item in artifact["acceptance_gate_results"].values():
            if item["passed"] is not None:
                item["passed"] = False
    artifact["qwen_pilot_complete_score"] = int(
        len(rows) == 72 and all(r["passed"] for r in receipts + terminal)
    )
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    custody.atomic_json(destination, artifact)
    progress(
        started,
        "publication",
        "after",
        path=destination,
        verdict=artifact["honest_verdict"],
        sha256=custody.sha256_file(destination),
    )
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: list[str] | None = None) -> int:
    """Accept the fixed production date or a read-only cold replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260926")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = cold_reduce(args.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    return run_experiment(ROOT, args.date, args.output)
