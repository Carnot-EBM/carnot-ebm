"""Exposed paired Qwen localization pilot. REQ-REPORT-7745."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Callable

from carnot import experiment_7729_v673_qwen_draft_pilot as prior
from carnot.reporting import current_work_receipt as custody
from carnot.reporting import experiment_7303_validation_scope as validation


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7745_v674_qwen_localization")
RESULT = Path("results/experiment_7745_v674_qwen_localization.json")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
ARMS = ("direct", "localized")
SEED = 67445
PRINCIPLE = "Completion and scientific benefit are separate; raw families bound the claim."
SCOPE = {
    "tests": ["tests/python/test_experiment_7745_v674_qwen_localization.py"],
    "changed_modules": ["python/carnot/experiment_7745_v674_qwen_localization.py"],
    "static_paths": ["scripts/experiments/experiment_7745_v674_qwen_localization.py"],
    "specs": ["REQ-REPORT-7745", "REQ-VERIFY-7745"],
    "e2e": ["task_owned_transport_both_arms", "task_cold_raw_replay"],
}


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush real work boundaries and current elapsed time."""
    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(
        f"[exp7745] {phase} {event} elapsed_s={time.monotonic() - started:.2f} {suffix}", flush=True
    )


def arm_order(family_id: str) -> tuple[str, str]:
    """Randomize the pair reproducibly within each independent family."""
    arms = list(ARMS)
    seed = int(hashlib.sha256(f"{SEED}:{family_id}".encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(arms)
    return arms[0], arms[1]


def make_request(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """Keep the original input and all decoding settings matched."""
    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    instruction = (
        "/no_think\nDecide whether the original answer contains unsupported content "
        "from the complete source. Return one JSON object. decision must be "
        "unsupported, supported, or abstain. quote must be one exact, unique "
        "substring of the source; a quote alone does not prove support. "
        "Abstain if the source does not permit a confident decision."
    )
    if arm == "localized":
        instruction += (
            " Include unsupported_spans as a JSON array. Each entry has text, "
            "an exact unique substring of the answer, and quote, an exact unique "
            "substring of the source linked to that span. Return [] if none."
        )
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": json.dumps(
                    {"complete_source": row["source"], "original_answer": row["answer"]},
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def score_response(row: dict[str, Any], arm: str, text: str, finish: str) -> dict[str, Any]:
    """Keep syntax, address, human decision and sentence truth separate."""
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        pass
    required = {"decision", "quote"} | ({"unsupported_spans"} if arm == "localized" else set())
    valid = (
        isinstance(parsed, dict)
        and set(parsed) == required
        and isinstance(parsed.get("decision"), str)
        and parsed.get("decision") in {"unsupported", "supported", "abstain"}
        and isinstance(parsed.get("quote"), str)
        and (arm != "localized" or isinstance(parsed.get("unsupported_spans"), list))
    )
    decision = parsed["decision"] if valid else None
    quote = parsed["quote"] if valid else None
    quote_valid = bool(quote and row["source"].count(quote) == 1)
    spans = parsed.get("unsupported_spans", []) if valid and arm == "localized" else []
    predicted: set[int] = set()
    span_valid = bool(valid)
    offsets = row.get("sentence_offsets", [])
    for item in spans:
        if not isinstance(item, dict) or set(item) != {"text", "quote"}:
            span_valid = False
            continue
        answer_text, linked = item["text"], item["quote"]
        if (
            not isinstance(answer_text, str)
            or not isinstance(linked, str)
            or not answer_text
            or row["answer"].count(answer_text) != 1
            or not linked
            or row["source"].count(linked) != 1
        ):
            span_valid = False
            continue
        start = row["answer"].index(answer_text)
        end = start + len(answer_text)
        predicted.update(
            i for i, (left, right) in enumerate(offsets) if start < right and end > left
        )
    truth = {i for i, label in enumerate(row.get("sentence_targets", [])) if label == 1}
    known = {i for i, label in enumerate(row.get("sentence_targets", [])) if label in {0, 1}}
    human = bool(row["annotation_types"]) if row["annotation_types"] is not None else None
    decided = decision in {"unsupported", "supported"} and finish == "stop"
    return {
        "syntax_valid": bool(valid),
        "decision": decision,
        "quote": quote,
        "quote_valid": quote_valid,
        "span_address_valid": span_valid if arm == "localized" else None,
        "unsupported_spans": spans,
        "semantic_verified": False,
        "unknown": not decided,
        "truncated": finish == "length",
        "censored": finish in {"length", "timeout", "error"},
        "human_binary_unsupported": human,
        "binary_accuracy": bool(
            decided and human is not None and (decision == "unsupported") == human
        ),
        "false_accept": bool(decided and human is True and decision == "supported"),
        "false_reject": bool(decided and human is False and decision == "unsupported"),
        "localization_tp": len(predicted & truth) if arm == "localized" else None,
        "localization_fp": len(predicted & (known - truth)) if arm == "localized" else None,
        "localization_fn": len(truth - predicted) if arm == "localized" else None,
        "localization_known_sentences": len(known) if arm == "localized" else None,
    }


def execute_family(
    row: dict[str, Any],
    transport: Callable[[dict[str, Any]], dict[str, Any] | bytes],
    raw_dir: Path,
    started: float,
) -> list[dict[str, Any]]:
    """Make exactly two current calls and retain their original bytes."""
    output = []
    for arm in arm_order(row["family_id"]):
        request = make_request(row, arm)
        stem = f"{row['family_id']}_{arm}"
        request_path = raw_dir / f"{stem}_request.json"
        response_path = raw_dir / f"{stem}_response.json"
        custody.atomic_json(request_path, request)
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
                answer if isinstance(answer, bytes) else json.dumps(answer, sort_keys=True).encode()
            )
            response_path.write_bytes(response_bytes)
            response = json.loads(response_bytes)
            choice = response["choices"][0]
            content = str(choice["message"].get("content") or "")
            finish = str(choice.get("finish_reason") or "unknown")
            usage = response.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            response_bytes = json.dumps(
                {"transport_error": f"{type(error).__name__}:{error}"}
            ).encode()
            response_path.write_bytes(response_bytes)
            content, finish, input_tokens, output_tokens = "", "error", 0, 0
        receipt = {
            "request_path": str(request_path),
            "request_sha256": custody.sha256_file(request_path),
            "raw_response_path": str(response_path),
            "raw_response_sha256": custody.sha256_file(response_path),
            "response_text": content,
            "finish_reason": finish,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": time.monotonic() - begin,
        }
        metrics = score_response(row, arm, content, finish)
        output.append(
            {
                "family_id": row["family_id"],
                "arm": arm,
                "official_split": row["official_split"],
                "prior_exposure": True,
                "source_sha256": hashlib.sha256(row["source"].encode()).hexdigest(),
                "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest(),
                "calls": [receipt],
                "metrics": metrics,
                "raw_metrics": metrics,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_s": receipt["latency_s"],
                "denominator": 1,
                "excluded": False,
                "exclusion_reason": None,
                "censored": metrics["censored"],
            }
        )
        progress(
            started,
            "generation",
            "after",
            family=row["family_id"],
            arm=arm,
            completed_units=len(output),
            output_tokens=output_tokens,
        )
    custody.atomic_json(raw_dir / f"checkpoint_{row['family_id']}.json", output)
    progress(started, "family", "after", family=row["family_id"], completed_units=len(output))
    return output


def reduce_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count families, not calls or sentences, and bootstrap paired deltas."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family_id"], {})[row["arm"]] = row
    complete = {name: arms for name, arms in grouped.items() if set(arms) == set(ARMS)}
    by_arm = {}
    for arm in ARMS:
        sample = [arms[arm] for arms in complete.values()]
        metrics = [row["metrics"] for row in sample]
        tp = sum(m["localization_tp"] or 0 for m in metrics)
        fp = sum(m["localization_fp"] or 0 for m in metrics)
        fn = sum(m["localization_fn"] or 0 for m in metrics)
        by_arm[arm] = {
            "denominator": len(sample),
            "binary_accuracy_numerator": sum(m["binary_accuracy"] for m in metrics),
            "false_accept_numerator": sum(m["false_accept"] for m in metrics),
            "false_reject_numerator": sum(m["false_reject"] for m in metrics),
            "unknown_numerator": sum(m["unknown"] for m in metrics),
            "syntax_valid_numerator": sum(m["syntax_valid"] for m in metrics),
            "quote_valid_numerator": sum(m["quote_valid"] for m in metrics),
            "span_address_valid_numerator": sum(bool(m["span_address_valid"]) for m in metrics),
            "truncated_numerator": sum(m["truncated"] for m in metrics),
            "localization_tp": tp,
            "localization_fp": fp,
            "localization_fn": fn,
            "localization_precision": tp / (tp + fp) if arm == "localized" and tp + fp else None,
            "localization_recall": tp / (tp + fn) if arm == "localized" and tp + fn else None,
            "input_tokens": sum(row["input_tokens"] for row in sample),
            "output_tokens": sum(row["output_tokens"] for row in sample),
            "latency_s": sum(row["latency_s"] for row in sample),
        }
        by_arm[arm]["binary_accuracy"] = (
            by_arm[arm]["binary_accuracy_numerator"] / len(sample) if sample else None
        )
        by_arm[arm]["unknown_rate"] = (
            by_arm[arm]["unknown_numerator"] / len(sample) if sample else None
        )
    deltas = [
        int(arms["localized"]["metrics"]["binary_accuracy"])
        - int(arms["direct"]["metrics"]["binary_accuracy"])
        for arms in complete.values()
    ]
    n = len(deltas)
    rng = random.Random(SEED)
    draws = (
        sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(2048))
        if n
        else []
    )
    return {
        "paired_families": n,
        "by_arm": by_arm,
        "comparison": {
            "paired_n": n,
            "binary_accuracy_delta": sum(deltas) / n if n else None,
            "paired_bootstrap_95_interval": [draws[51], draws[1996]] if n else None,
            "per_family_deltas": dict(zip(complete, deltas)),
        },
    }


def cold_reduce_rows(rows: list[dict[str, Any]], panel_path: Path) -> dict[str, Any]:
    """Reopen exact source, request, response, and counters in a fresh reader."""
    panel = {row["family_id"]: row for row in json.loads(panel_path.read_text())}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["family_id"], row["arm"])
        if key in seen or key[0] not in panel or key[1] not in ARMS or len(row["calls"]) != 1:
            return {"passed": False, "reason": "family_arm_identity"}
        seen.add(key)
        original = panel[key[0]]
        call = row["calls"][0]
        if (
            row["source_sha256"] != hashlib.sha256(original["source"].encode()).hexdigest()
            or row["answer_sha256"] != hashlib.sha256(original["answer"].encode()).hexdigest()
            or row["denominator"] != 1
            or row["excluded"]
            or row["input_tokens"] != call["input_tokens"]
            or row["output_tokens"] != call["output_tokens"]
            or row["latency_s"] != call["latency_s"]
        ):
            return {"passed": False, "reason": "row_identity_or_totals"}
        request_path, response_path = Path(call["request_path"]), Path(call["raw_response_path"])
        if (
            custody.sha256_file(request_path) != call["request_sha256"]
            or custody.sha256_file(response_path) != call["raw_response_sha256"]
        ):
            return {"passed": False, "reason": "raw_hash"}
        if json.loads(request_path.read_text()) != make_request(original, key[1]):
            return {"passed": False, "reason": "request_mismatch"}
        response = json.loads(response_path.read_bytes())
        if "transport_error" in response:
            content, finish, input_tokens, output_tokens = "", "error", 0, 0
        else:
            choice = response["choices"][0]
            content = str(choice["message"].get("content") or "")
            finish = str(choice.get("finish_reason") or "unknown")
            usage = response.get("usage") or {}
            input_tokens, output_tokens = (
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
            )
        if (
            content != call["response_text"]
            or finish != call["finish_reason"]
            or input_tokens != call["input_tokens"]
            or output_tokens != call["output_tokens"]
            or output_tokens > 256
            or row["metrics"] != score_response(original, key[1], content, finish)
        ):
            return {"passed": False, "reason": "metric_or_token_mismatch"}
    return {"passed": len(seen) == len(panel) * 2, "families": len(panel), "calls": len(seen)}


def prepare_panel(
    root: Path, started: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate current model, original release, and V673 family identity."""
    from carnot.inference.sota_models import cached_current_model
    from carnot.experiment_7423_v651_annotated_protocol import (
        authenticate_assets,
        load_release,
        DEFAULT_CACHE_ROOT,
    )
    from carnot.experiment_7740_v674_sentence_label_protocol import map_targets

    checks, hashes, context = prior.preflight(root, started)
    current = cached_current_model()
    selected_path = str(context.get("model_path") or "")
    cache_path = str((current or {}).get("model_path") or "")
    checks.append(
        prior.gate(
            "cached_current_model",
            "local_model_cache",
            Path(cache_path or "/missing-qwen.gguf"),
            "current_model_path",
            selected_path,
            cache_path,
        )
    )
    historical = root / prior.RESULT
    hashes["flagged_historical_evidence"][str(prior.RESULT)] = (
        custody.sha256_file(historical) if historical.is_file() else None
    )
    if any(not check["passed"] for check in checks):
        return [], checks, hashes, context
    panel, _ = prior.prior.freeze_panel(root, started)
    previous = root / "results/raw/experiment_7729_v673_qwen_draft_pilot/frozen_panel.json"
    checks.append(
        prior.gate(
            "v673_panel_bytes",
            "exp7729",
            previous,
            "same_exposed_source_and_answer_bytes",
            True,
            previous.is_file() and json.loads(previous.read_text()) == panel,
        )
    )
    hashes["valid_producers"][str(previous.relative_to(root))] = (
        custody.sha256_file(previous) if previous.is_file() else None
    )
    receipt = authenticate_assets(DEFAULT_CACHE_ROOT)
    _, responses = load_release(receipt, started=started)
    labels = {response["id"]: response.get("labels") for response in responses}
    for row in panel:
        mapped = map_targets(row["answer"].encode(), labels.get(row["response_id"]))
        row["sentence_targets"] = mapped["targets"]
        row["sentence_offsets"] = mapped["char_offsets"]
        row["sentence_target_reason"] = mapped["reason"]
    checks.append(
        prior.gate(
            "exposed_panel_size",
            "exp7729",
            previous,
            "distinct_families",
            24,
            len({row["family_id"] for row in panel}),
        )
    )
    return panel, checks, hashes, context


def build_artifact(
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    runtime: dict[str, Any],
    spans: list[dict[str, Any]],
    duration: float,
    date: str,
) -> dict[str, Any]:
    """State completion without promoting the exposed diagnostic pilot."""
    paired = reduce_pairs(rows)
    failures = [check for check in checks if not check["passed"]]
    calls = [call for row in rows for call in row["calls"]]
    complete = len(rows) == 48 and paired["paired_families"] == 24 and len(calls) == 48
    invoked = bool(runtime.get("model_load_attempted"))
    blocked = bool(failures and not invoked)
    verdict_class = "blocked" if blocked else "null" if complete and not failures else "partial"
    verdict = (
        "complete_blocked_" + failures[0]["check"]
        if blocked
        else "complete_null_exposed_localization_pilot"
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
    gates["validity"].update(
        passed=not failures, measured_operands={"failed_checks": len(failures)}
    )
    gates["readiness"].update(
        passed=False,
        measured_operands={
            "paired_families": paired["paired_families"],
            "required_terminal_checks_pending": True,
        },
    )
    gates["decision_benefit"]["measured_operands"] = paired["comparison"]
    gates["efficiency"]["measured_operands"] = {
        "input_tokens": sum(call["input_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "latency_s": sum(call["latency_s"] for call in calls),
    }
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7745.v674.qwen_localization.v1",
        "experiment_id": "exp7745-qwen-localization",
        "milestone": "2026.09.674",
        "run_date": date,
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
            "started_families": len({row["family_id"] for row in rows}),
            "completed_families": paired["paired_families"],
            "eligible_families": 24 if rows else 0,
            "excluded_families": 0,
            "censored_families": len({row["family_id"] for row in rows if row["censored"]}),
            "effective_independent_n": paired["paired_families"],
            "intended_calls": 48,
            "started_calls": runtime.get("generation_attempted", 0),
            "completed_calls": len(calls),
            "output_token_ceiling": 12288,
            "output_tokens": sum(call["output_tokens"] for call in calls),
            "arms_and_sentences_are_not_independent_families": True,
        },
        "claim_scope": "development_only",
        "fresh_generalization_eligible": False,
        "inference_substrate": "owned_local_llama_cpp_bounded_generation"
        if runtime.get("model_load_completed")
        else "no_model_load",
        "inference_substrate_class": "model_bounded_generation"
        if runtime.get("generation_attempted")
        else "no_model_load",
        "planned_inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": [MODEL_ID] if invoked else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_specs": [
            {
                "hf_id": MODEL_ID,
                "model_path": runtime.get("model_path"),
                "model_sha256": runtime.get("model_sha256"),
                "quantization": "Q4_K_M",
            }
        ]
        if invoked
        else [],
        "model_invoked": invoked,
        "invocation_counts": {
            "loads": runtime.get("model_load_attempted", 0),
            "forwards": runtime.get("generation_attempted", 0),
            "generations": len(calls),
            "input_tokens": sum(call["input_tokens"] for call in calls),
            "output_tokens": sum(call["output_tokens"] for call in calls),
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
            "arm_order": "sha256:67445:family_id",
            "bootstrap": SEED,
        },
        "reproducibility_checksum": custody.canonical_hash(
            {
                "source_hashes": hashes,
                "seed": SEED,
                "arms": ARMS,
                "max_tokens": 256,
                "reducer_sha256": custody.sha256_file(Path(__file__)),
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
        "verifier_is_oracle": False,
        "qwen_localization_complete_score": 0,
        "current_model_receipts": runtime,
        "activation": False,
        "production_promotion": False,
    }
    artifact["field_principles"] = {
        **{key: PRINCIPLE for key in artifact},
        **{f"acceptance_gate_{key}": PRINCIPLE for key in gates},
    }
    return artifact


def owned_capture(
    root: Path,
    panel: list[dict[str, Any]],
    context: dict[str, Any],
    started: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Launch the established task-owned CUDA server and release only it."""
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
        task_id="experiment_7745_v674_qwen_localization",
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
        "generation_failures": 0,
        "cancellations": 0,
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
            proposer._ensure_server, started=started, phase="exp7745_model_load"
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
            log_path,
            owned_vram,
            _observed_offload_layers(log_path),
        )
        runtime["runtime_build"] = v664._runtime_build_receipt()
        if not runtime["offload_layers"].get("actual_offload"):
            raise RuntimeError("qwen_gpu_offload_not_authenticated")
        observed_template = str(runtime["server_props"].get("chat_template") or "")
        if (
            hashlib.sha256(observed_template.encode()).hexdigest()
            != context["chat_template_sha256"]
        ):
            raise RuntimeError("chat_template_changed_after_load")
        lease.transition("resident", vram_mb=int(owned_vram))
        lease.transition("inferencing")
        run_dir = root / RAW / "runs" / f"{int(time.time())}-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        runtime["run_dir"] = str(run_dir)

        def transport(request: dict[str, Any]) -> bytes:
            runtime["generation_attempted"] += 1
            return _call_with_heartbeats(
                lambda: v664._post_json(proposer._url() + "/v1/chat/completions", request, 900),
                started=started,
                phase="exp7745_generation",
            )

        for index, family in enumerate(panel, 1):
            rows.extend(execute_family(family, transport, run_dir, started))
            runtime["generation_failures"] = sum(
                call["finish_reason"] == "error" for row in rows for call in row["calls"]
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
            )
    finally:
        progress(started, "model_unload", "before")
        if proposer is not None:
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


def phase_span(
    name: str, started: float, begin: float, units: int, checkpoint: Path, date: str
) -> dict[str, Any]:
    """Keep disjoint monotonic time and exact checkpoint identity."""
    end = time.monotonic()
    return {
        "phase": name,
        "start_s": begin - started,
        "end_s": end - started,
        "duration_s": end - begin,
        "run_date": date,
        "heartbeat_times_s": [end - started],
        "completed_units": units,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": custody.sha256_file(checkpoint) if checkpoint.is_file() else None,
    }


def cold_reduce(path: Path, raw_dir: Path | None = None) -> dict[str, Any]:
    """Recompute final rows from frozen input and saved transport bytes."""
    artifact = json.loads(path.read_text())
    rows = artifact["rows"]
    if artifact["verdict_class"] == "blocked":
        return {"passed": not rows and not artifact["model_invoked"], "families": 0, "calls": 0}
    panel_path = (raw_dir or ROOT / RAW) / "frozen_panel.json"
    receipt = cold_reduce_rows(rows, panel_path)
    if receipt["passed"] and reduce_pairs(rows) != artifact["paired_family_results"]:
        return {"passed": False, "reason": "paired_reduction_mismatch"}
    return receipt


def terminal_commands(root: Path, candidate: Path) -> list[validation.CommandSpec]:
    """Register fresh reader processes on the same candidate bytes."""
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
    """Prequalify, capture once, independently read, and publish atomically."""
    started = time.monotonic()
    progress(started, "startup", "before", root=root.resolve())
    root = root.resolve(strict=True)
    if date != "20260927":
        raise ValueError("date_must_be_20260927")
    destination = output if output.is_absolute() else root / output
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7745-"))
    (private / "pytest").mkdir(parents=True, exist_ok=True)
    scope_path = raw_dir / "frozen_affected_scope.json"
    custody.atomic_json(scope_path, SCOPE)
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    progress(started, "preconditions", "before")
    try:
        panel, checks, hashes, context = prepare_panel(root, started)
    except (OSError, ValueError, KeyError, TypeError) as error:
        panel, context = [], {}
        checks = [
            prior.gate(
                "source_reconstruction",
                "RAGTruth",
                root / prior.RAW,
                "authenticated_original_panel",
                True,
                f"{type(error).__name__}:{error}",
            )
        ]
        hashes = {
            "valid_producers": {},
            "flagged_historical_evidence": {},
            "pre_gate_receipts": {},
            "missing_custody": [str(root / prior.RAW)],
        }
    panel_path = raw_dir / "frozen_panel.json"
    if panel and all(check["passed"] for check in checks):
        custody.atomic_json(panel_path, panel)
        hashes["pre_gate_receipts"][str(RAW / "frozen_panel.json")] = custody.sha256_file(
            panel_path
        )
    spans.append(phase_span("preconditions", started, begin, len(checks), panel_path, date))
    failures = [check for check in checks if not check["passed"]]
    progress(started, "preconditions", "after", completed_units=len(checks), failures=len(failures))

    begin = time.monotonic()
    progress(started, "scoped_validation", "before")
    commands = validation.build_scoped_commands(
        root,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=private / "pytest",
        coverage_file=private / ".coverage.exp7745",
    )
    receipts = validation.run_commands(
        root, commands, log_dir=raw_dir / "validation" / "affected", heartbeat_s=45
    )
    spans.append(phase_span("scoped_validation", started, begin, len(receipts), scope_path, date))
    valid_checks = all(receipt["passed"] for receipt in receipts)
    progress(
        started, "scoped_validation", "after", completed_units=len(receipts), passed=valid_checks
    )

    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    if not failures and valid_checks:
        begin = time.monotonic()
        progress(started, "measurement", "before", families=len(panel), max_calls=48)
        try:
            rows, runtime = owned_capture(root, panel, context, started)
        except BaseException as error:
            runs = sorted((raw_dir / "runs").glob(f"*-{os.getpid()}"))
            checkpoint = runs[-1] / "checkpoint.json" if runs else None
            if checkpoint and checkpoint.is_file():
                rows = json.loads(checkpoint.read_text())["rows"]
            runtime = {
                "model_load_attempted": 1,
                "generation_attempted": len(rows),
                "model_path": str(context["model_path"]),
                "model_sha256": context["model_sha256"],
                "device_uuid": context["selected"]["uuid"],
                "error": f"{type(error).__name__}:{error}",
            }
            checks.append(
                prior.gate(
                    "owned_capture",
                    "exp7630_owned_qwen_server",
                    raw_dir,
                    "paired_families",
                    24,
                    len({row["family_id"] for row in rows}),
                )
            )
        spans.append(phase_span("measurement", started, begin, len(rows), raw_dir / "runs", date))
        progress(
            started,
            "measurement",
            "after",
            completed_units=len(rows),
            failures=len([check for check in checks if not check["passed"]]),
        )
    rows_path = raw_dir / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    hashes["pre_gate_receipts"][str(RAW / "rows.jsonl")] = custody.sha256_file(rows_path)
    artifact = build_artifact(
        rows, checks, hashes, runtime, spans, time.monotonic() - started, date
    )
    artifact["validation_receipts"]["required_commands"] = receipts
    if not valid_checks:
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
    spans.append(phase_span("terminal_readers", started, begin, len(terminal), candidate, date))
    progress(
        started,
        "terminal_readers",
        "after",
        completed_units=len(terminal),
        passed=all(item["passed"] for item in terminal),
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["validation_receipts"]["terminal_candidate_sha256"] = custody.sha256_file(candidate)
    artifact["flagged_adversarial"] = not next(
        item["passed"] for item in terminal if item["name"] == "adversarial_verify"
    )
    all_valid = valid_checks and all(item["passed"] for item in terminal)
    if not all_valid:
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
    complete = artifact["paired_family_results"]["paired_families"] == 24 and len(rows) == 48
    artifact["qwen_localization_complete_score"] = int(complete and all_valid and not failures)
    artifact["acceptance_gate_results"]["readiness"].update(
        passed=bool(artifact["qwen_localization_complete_score"]),
        measured_operands={
            "paired_families": artifact["paired_family_results"]["paired_families"],
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
        path=destination,
        verdict=artifact["honest_verdict"],
        sha256=custody.sha256_file(destination),
    )
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: list[str] | None = None) -> int:
    """Accept the fixed production date or a read-only cold replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = cold_reduce(args.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    return run_experiment(ROOT, args.date, args.output)
