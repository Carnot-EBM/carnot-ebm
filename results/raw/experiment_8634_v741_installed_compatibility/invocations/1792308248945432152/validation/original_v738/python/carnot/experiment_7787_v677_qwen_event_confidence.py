"""Paired Qwen event-probability pilot. REQ-REPORT-7787, REQ-VERIFY-7787."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable

from carnot import experiment_7745_v674_qwen_localization as source
from carnot import experiment_7770_v676_qwen_runner_qualification as transport
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7787_v677_qwen_event_confidence")
RESULT = Path("results/experiment_7787_v677_qwen_event_confidence.json")
PANEL = Path("results/raw/experiment_7745_v674_qwen_localization/frozen_panel.json")
PRODUCER = Path("results/experiment_7745_v674_qwen_localization.json")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
SEED = 67701
ARMS = ("generic", "event")
PROMPTS = {
    "generic": "/no_think\nGive the probability that the original answer is correct given the complete source.",
    "event": "/no_think\nGive the probability that at least one claim in the original answer lacks support in the complete source.",
}
FIELD_PRINCIPLES = {
    "experiment_id": "Each record needs a unique owner.",
    "honest_verdict": "Unchanged external inputs must not consume retries.",
    "verdict_class": "Claim strength travels with the result.",
    "flagged_adversarial": "Invalid evidence cannot open a downstream gate.",
    "gate_check_summary": "A missing producer differs from a scientific null.",
    "rows": "A headline must be reproducible from individual units.",
    "acceptance_gate_results": "Working fixtures do not establish benefit.",
    "duration_s": "Real elapsed work determines substrate authenticity.",
    "phase_spans": "Measured work must cover validation and cold replay.",
    "random_seed": "Another process must recover the same inputs.",
    "reproducibility_checksum": "Exact code, data, roles, parameters and seed must bind.",
    "sample_size_budget": "Views do not create independent families.",
    "source_artifact_hashes": "An old file cannot replace a missing current producer.",
    "preconditions_checked": "Cheap failures precede expensive work.",
    "validation_receipts": "Every required check must pass before readiness.",
    "verifier_is_oracle": "Fixture truth is circular evidence.",
    "claim_scope": "Natural data remain exposed development evidence.",
    "inference_substrate": "Duration floors follow invoked work.",
    "inference_substrate_class": "Planned and actual work must be distinct.",
    "MODEL_SPECS": "Only current invocations belong in model metadata.",
    "model_specs": "Actual loaded bytes determine model identity.",
    "model_invocation_counts": "A call count must describe current calls.",
    "qwen_protocol_ready_score": "New scoped readiness cannot rewrite old readiness.",
    "parse_coverage_by_arm": "Grammar success is separate from calibration.",
    "semantic_comparison_rows": "Natural labels belong only in the evaluator.",
    "model_file_sha256": "A cache name alone is not provenance.",
    "gpu_offload_receipt": "Live GPU requires actual offload evidence.",
    "historical_full_suite_status": "The old failed contract remains failed.",
}


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Flush each work boundary so an owned child remains observable."""
    print(
        f"[exp7787] phase={phase} event={event} elapsed_s={time.monotonic() - start:.2f} completed_units={units}",
        flush=True,
    )


def digest(value: str) -> str:
    """Hash exact UTF-8 bytes so a reconstructed panel cannot silently drift."""
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def make_protocol(panel: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze all model-visible roles and decode settings before observation."""
    base = json.loads((ROOT / transport.PROTOCOL).read_text())
    request = dict(base["request"])
    request["seed"] = SEED
    return {
        "schema": "carnot.exp7787.v677.protocol.v1",
        "seed": SEED,
        "family_ids": [row["family_id"] for row in panel],
        "input_hashes": {
            row["family_id"]: {
                "source_sha256": digest(row["source"]),
                "answer_sha256": digest(row["answer"]),
            }
            for row in panel
        },
        "prompts": PROMPTS,
        "common_instruction": base["common_instruction"],
        "request": request,
        "arm_order": list(ARMS),
    }


def make_request(protocol: dict[str, Any], family: dict[str, Any], arm: str) -> dict[str, Any]:
    """Reuse the tested request builder with the new frozen protocol."""
    return transport.make_request(protocol, family, arm)


def parse_probability(text: str, finish: str, arm: str) -> dict[str, Any]:
    """Reuse the tested strict parser and invalid-row risk convention."""
    return transport.parse_probability(text, finish, arm)


def gate(
    name: str, upstream: str, path: Path, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep both operands and the supplying file in a failed check."""
    return {
        "check": name,
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path) if path.is_file() else None,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Authenticate the science producer before accepting a pre-gate panel."""
    producer, panel_path = root / PRODUCER, root / PANEL
    checks = [
        gate("producer_exists", "exp7745", producer, "exists", True, producer.is_file()),
        gate("panel_exists", "exp7745_pre_gate", panel_path, "exists", True, panel_path.is_file()),
    ]
    hashes = {
        "producer": {
            "path": str(PRODUCER),
            "sha256": sha256_file(producer) if producer.is_file() else None,
            "date": "20260927",
            "imported_fields": [
                "verdict_class",
                "qwen_localization_complete_score",
                "source_artifact_hashes",
            ],
            "eligible": False,
        },
        "panel": {
            "path": str(PANEL),
            "sha256": sha256_file(panel_path) if panel_path.is_file() else None,
            "date": "20260927",
            "imported_fields": ["family_id", "source", "answer", "annotation_types"],
            "eligible": False,
        },
    }
    if any(not check["passed"] for check in checks):
        return [], checks, hashes
    old = json.loads(producer.read_text())
    panel = json.loads(panel_path.read_text())
    for field, expected in (
        ("milestone", "2026.09.674"),
        ("verdict_class", "null"),
        ("qwen_localization_complete_score", 1),
        ("flagged_adversarial", False),
    ):
        checks.append(gate("producer_field", "exp7745", producer, field, expected, old.get(field)))
    expected_hash = (
        old.get("source_artifact_hashes", {}).get("pre_gate_receipts", {}).get(str(PANEL))
    )
    checks.append(
        gate(
            "panel_hash",
            "exp7745_pre_gate",
            panel_path,
            "sha256",
            expected_hash,
            sha256_file(panel_path),
        )
    )
    checks.append(
        gate(
            "family_count",
            "exp7745_pre_gate",
            panel_path,
            "independent_families",
            24,
            len({row["family_id"] for row in panel}),
        )
    )
    checks.append(
        gate(
            "producer_family_count",
            "exp7745",
            producer,
            "paired_family_results.paired_families",
            24,
            old.get("paired_family_results", {}).get("paired_families"),
        )
    )
    hashes["producer"]["eligible"] = all(c["passed"] for c in checks)
    hashes["panel"]["eligible"] = hashes["producer"]["eligible"]
    return (panel if hashes["producer"]["eligible"] else []), checks, hashes


def score_row(
    family: dict[str, Any],
    arm: str,
    parsed: dict[str, Any],
    usage: dict[str, Any],
    latency_s: float,
) -> dict[str, Any]:
    """Join natural labels only after parsing, then price each decision."""
    truth = bool(family["annotation_types"]) if family.get("annotation_types") is not None else None
    risk = parsed["unsupported_risk"]
    escalate = bool(parsed["forced_escalation"] or risk >= 0.5)
    return {
        "family_id": family["family_id"],
        "arm": arm,
        "source_sha256": digest(family["source"]),
        "answer_sha256": digest(family["answer"]),
        "prior_exposure": True,
        "metrics": parsed,
        "natural_unsupported": truth,
        "escalated": escalate,
        "false_accept": bool(truth is True and not escalate),
        "brier": (risk - int(truth)) ** 2 if truth is not None else None,
        "decision_cost": (1 if escalate else 5 if truth else 0) if truth is not None else None,
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "latency_s": latency_s,
        "disposition": "completed" if parsed["valid"] else "rejected",
        "censored": parsed["finish_reason"] in {"length", "timeout", "error", "missing"},
        "excluded": False,
    }


def reduce_rows(rows: list[dict[str, Any]], seed: int = SEED) -> dict[str, Any]:
    """Bootstrap paired family differences while retaining invalid syntax."""
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = row["family_id"], row["arm"]
        if key in seen:
            raise ValueError("duplicate_pair")
        seen.add(key)
        if row["arm"] in ARMS and row["disposition"] != "unstarted":
            grouped.setdefault(row["family_id"], {})[row["arm"]] = row
    pairs = [group for group in grouped.values() if set(group) == set(ARMS)]
    arm_rows = {arm: [pair[arm] for pair in pairs] for arm in ARMS}
    coverage = {
        arm: sum(row["metrics"]["valid"] for row in arm_rows[arm]) / len(pairs) if pairs else None
        for arm in ARMS
    }
    semantic = {}
    for arm in ARMS:
        measured = [row for row in arm_rows[arm] if row["natural_unsupported"] is not None]
        valid = [row for row in measured if row["metrics"]["valid"]]
        semantic[arm] = {
            "families": len(measured),
            "false_accepts": sum(row["false_accept"] for row in measured),
            "mean_brier": sum(row["brier"] for row in measured) / len(measured)
            if measured
            else None,
            "valid_only_brier": sum(row["brier"] for row in valid) / len(valid) if valid else None,
            "mean_cost": sum(row["decision_cost"] for row in measured) / len(measured)
            if measured
            else None,
            "parse_valid": sum(row["metrics"]["valid"] for row in arm_rows[arm]),
            "denominator": len(pairs),
        }
    paired = [
        pair for pair in pairs if all(pair[arm]["natural_unsupported"] is not None for arm in ARMS)
    ]
    rng = random.Random(seed)
    improvements = {}
    for key, field in (("brier", "brier"), ("cost", "decision_cost")):
        deltas = [pair["generic"][field] - pair["event"][field] for pair in paired]
        n = len(deltas)
        draws = (
            sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(2048))
            if n
            else []
        )
        improvements[key] = {
            "mean": sum(deltas) / n if n else None,
            "lower95": draws[51] if n else None,
            "upper95": draws[1996] if n else None,
            "per_family": {
                pair["generic"]["family_id"]: delta for pair, delta in zip(paired, deltas)
            },
        }
    passed = bool(
        len(pairs) == 24
        and all(coverage[arm] is not None and coverage[arm] >= 0.9 for arm in ARMS)
        and all(
            improvements[key]["lower95"] is not None and improvements[key]["lower95"] > 0
            for key in ("brier", "cost")
        )
        and semantic["event"]["false_accepts"] <= semantic["generic"]["false_accepts"]
    )
    return {
        "independent_n": len(pairs),
        "parse_coverage_by_arm": coverage,
        "semantic_comparison_rows": semantic,
        "paired_improvements": improvements,
        "benefit_passed": passed,
    }


def capture_family(
    family: dict[str, Any],
    protocol: dict[str, Any],
    call: Callable[[dict[str, Any]], dict[str, Any]],
    raw_dir: Path,
    started: float,
) -> list[dict[str, Any]]:
    """Save original request and response bytes for each bounded call."""
    rows = []
    for arm in ARMS:
        request = make_request(protocol, family, arm)
        request["stream"] = False
        stem = f"{family['family_id']}_{arm}"
        request_path, response_path = (
            raw_dir / f"{stem}_request.json",
            raw_dir / f"{stem}_response.json",
        )
        atomic_json(request_path, request)
        progress(started, "generation", "before", len(rows))
        begin = time.monotonic()
        try:
            response = call(request)
            if response.get("model") not in {MODEL_ID, protocol.get("served_model_id")}:
                raise ValueError("foreign_response_model")
            choice = response["choices"][0]
            text = str(choice["message"].get("content") or "")
            finish = str(choice.get("finish_reason") or "missing")
            usage = response.get("usage") or {}
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            response = {"transport_error": f"{type(error).__name__}:{error}"}
            text, finish, usage = "", "error", {}
        atomic_json(response_path, response)
        row = score_row(
            family, arm, parse_probability(text, finish, arm), usage, time.monotonic() - begin
        )
        row.update(
            {
                "raw_request_path": str(request_path),
                "raw_request_sha256": sha256_file(request_path),
                "raw_response_path": str(response_path),
                "raw_response_sha256": sha256_file(response_path),
                "response_text": text,
                "finish_reason": finish,
            }
        )
        rows.append(row)
        progress(started, "generation", "after", len(rows))
    atomic_json(raw_dir / f"checkpoint_{family['family_id']}.json", rows)
    return rows


def cold_reduce(path: Path) -> dict[str, Any]:
    """Reopen frozen source, request and response bytes in a fresh reader."""
    value = json.loads(path.read_text())
    panel = {item["family_id"]: item for item in value["panel"]}
    rows = value["rows"]
    for row in rows:
        if row.get("disposition") == "unstarted":
            continue
        family = panel[row["family_id"]]
        req, reply = Path(row["raw_request_path"]), Path(row["raw_response_path"])
        if sha256_file(req) != row["raw_request_sha256"]:
            raise ValueError("request_hash_changed")
        if sha256_file(reply) != row["raw_response_sha256"]:
            raise ValueError("response_hash_changed")
        request = make_request(value["protocol"], family, row["arm"])
        request["stream"] = False
        if json.loads(req.read_text()) != request:
            raise ValueError("request_changed")
        response = json.loads(reply.read_text())
        if "transport_error" in response:
            text, finish, usage = "", "error", {}
        else:
            choice = response["choices"][0]
            text, finish, usage = (
                str(choice["message"].get("content") or ""),
                str(choice.get("finish_reason") or "missing"),
                response.get("usage") or {},
            )
        expected = score_row(
            family, row["arm"], parse_probability(text, finish, row["arm"]), usage, row["latency_s"]
        )
        if (
            any(row[key] != expected[key] for key in expected)
            or text != row["response_text"]
            or finish != row["finish_reason"]
        ):
            raise ValueError("row_changed")
    reduced = reduce_rows(rows)
    if "reduced" in value and value["reduced"] != reduced:
        raise ValueError("reduction_changed")
    return reduced


def owned_capture(
    root: Path,
    panel: list[dict[str, Any]],
    protocol: dict[str, Any],
    context: dict[str, Any],
    started: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Launch only the selected GGUF server and retain actual CUDA evidence."""
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.experiment_7431_v651_arc_live_sentinel import _free_port
    from carnot.experiment_7581_v662_arc_bounded_canary import (
        _call_with_heartbeats,
        _observed_offload_layers,
        _owned_vram_mb,
    )
    from carnot.experiment_7604_v664_evidence_pilot import (
        _offload_receipt,
        _post_json,
        _runtime_build_receipt,
    )

    selected = context["selected"]
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
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
        timeout=90,
        tries=1,
        extra_server_args=("-lv", "4"),
    )
    proposer.model_repository = MODEL_ID
    proposer.requested_model_path = str(context["model_path"])
    runtime: dict[str, Any] = {
        "model_load_attempted": 1,
        "model_load_completed": 0,
        "canary_calls": 0,
        "panel_calls": 0,
        "model_path": str(context["model_path"]),
        "model_sha256": context["model_sha256"],
        "selected_gpu": selected,
        "backend": "llama.cpp CUDA",
    }
    rows: list[dict[str, Any]] = []
    try:
        progress(started, "model_load", "before")
        load_start = time.monotonic()
        healthy = _call_with_heartbeats(
            proposer._ensure_server, started=started, phase="exp7787_model_load"
        )
        runtime["model_load_s"] = time.monotonic() - load_start
        progress(started, "model_load", "after", int(bool(healthy)))
        if not healthy or not proposer._proc:
            raise RuntimeError("owned_qwen_load_failed")
        runtime["model_load_completed"] = 1
        runtime["server_pid"] = proposer._proc.pid
        props = proposer.server_props()
        runtime["server_props"] = props
        if (
            not props
            or hashlib.sha256(str(props.get("chat_template") or "").encode()).hexdigest()
            != context["chat_template_sha256"]
        ):
            raise RuntimeError("chat_template_mismatch")
        log_path = Path(proposer._stderr_log_path) if proposer._stderr_log_path else None
        memory = _owned_vram_mb(proposer._proc.pid)
        runtime["owned_vram_mb"] = memory
        runtime["gpu_offload_receipt"] = _offload_receipt(
            log_path, memory, _observed_offload_layers(log_path)
        )
        runtime["runtime_build"] = _runtime_build_receipt()
        if not runtime["gpu_offload_receipt"].get("actual_offload"):
            raise RuntimeError("qwen_gpu_offload_not_authenticated")
        from urllib import request as urlrequest

        with urlrequest.urlopen(proposer._url() + "/v1/models", timeout=3) as response:
            roster = json.load(response)
        ids = [item["id"] for item in roster["data"]]
        if (
            len(ids) != 1
            or Path(ids[0]).name not in {Path(context["model_path"]).name, MODEL_ID.split("/")[-1]}
            and ids[0] != MODEL_ID
        ):
            raise RuntimeError("wrong_model_server")
        protocol["served_model_id"] = ids[0]
        runtime["server_model_id"] = ids[0]

        def call(request: dict[str, Any]) -> dict[str, Any]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("capture_deadline")
            raw = _call_with_heartbeats(
                lambda: _post_json(
                    proposer._url() + "/v1/chat/completions", request, min(90.0, remaining)
                ),
                started=started,
                phase="exp7787_generation",
            )
            return json.loads(raw)

        for arm in ARMS:
            canary = make_request(
                protocol, {"source": "The sky is blue.", "answer": "The sky is blue."}, arm
            )
            canary["max_tokens"] = 32
            canary["stream"] = False
            progress(started, "canary", "before", runtime["canary_calls"])
            deadline = time.monotonic() + 90
            reply = call(canary)
            runtime["canary_calls"] += 1
            choice = reply["choices"][0]
            parsed = parse_probability(
                str(choice["message"].get("content") or ""),
                str(choice.get("finish_reason") or "missing"),
                arm,
            )
            runtime.setdefault("canaries", []).append(
                {"arm": arm, "request": canary, "response": reply, "parsed": parsed}
            )
            progress(started, "canary", "after", runtime["canary_calls"])
            if not parsed["valid"]:
                raise RuntimeError("schema_canary_failed")
        deadline = time.monotonic() + 1800
        run_dir = root / RAW / "runs" / f"{int(time.time())}-{os.getpid()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        runtime["run_dir"] = str(run_dir)
        for index, family in enumerate(panel):
            if time.monotonic() >= deadline:
                break
            pair = capture_family(family, protocol, call, run_dir, started)
            rows.extend(pair)
            runtime["panel_calls"] += len(pair)
            atomic_json(
                run_dir / "checkpoint.json", {"rows": rows, "completed_families": index + 1}
            )
            progress(started, "measurement", "checkpoint", index + 1)
        runtime["capture_s"] = 1800 - max(0.0, deadline - time.monotonic())
    finally:
        progress(started, "model_unload", "before", len(rows))
        proposer.stop()
        progress(started, "model_unload", "after", len(rows))
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    return rows, runtime


def build_artifact(
    panel: list[dict[str, Any]],
    protocol: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    hashes: dict[str, Any],
    runtime: dict[str, Any],
    receipts: dict[str, Any],
    started: float,
    spans: list[dict[str, Any]],
    date: str,
) -> dict[str, Any]:
    """Keep every planned unit and separate protocol validity from benefit."""
    all_rows = list(rows)
    used = {(row["family_id"], row["arm"]) for row in rows}
    for family in panel:
        for arm in ARMS:
            if (family["family_id"], arm) not in used:
                all_rows.append(
                    {
                        "family_id": family["family_id"],
                        "arm": arm,
                        "disposition": "unstarted",
                        "source_sha256": digest(family["source"]),
                        "answer_sha256": digest(family["answer"]),
                        "metrics": None,
                        "natural_unsupported": None,
                        "excluded": False,
                        "censored": True,
                    }
                )
    reduced = reduce_rows(all_rows)
    failed_checks = [item for item in checks if not item["passed"]]
    required = receipts.get("required", [])
    required_passed = all(item.get("passed") for item in required) and bool(required)
    complete = (
        len(rows) == 48
        and runtime.get("model_load_completed") == 1
        and runtime.get("gpu_offload_receipt", {}).get("actual_offload") is True
    )
    if failed_checks:
        verdict, cls = "complete_blocked_required_producer", "blocked"
    elif not required_passed or not complete:
        verdict, cls = "complete_disqualified_required_checks", "disqualified"
    elif reduced["benefit_passed"]:
        verdict, cls = "complete_positive_exposed_pilot", "positive"
    else:
        verdict, cls = "complete_null_exposed_pilot", "null"
    ready = int(complete and required_passed and not failed_checks)
    gates = {
        "validity": {
            "passed": not failed_checks,
            "measured_operands": {"failed_checks": len(failed_checks)},
            "principle": "Custody precedes measurement.",
        },
        "readiness": {
            "passed": bool(ready),
            "measured_operands": {
                "complete_calls": len(rows),
                "required_checks_passed": required_passed,
            },
            "principle": "An invalid check cannot qualify capture.",
        },
        "probability_quality": {
            "passed": reduced["benefit_passed"] if complete else None,
            "measured_operands": reduced["semantic_comparison_rows"] if complete else {},
            "principle": "Grammar does not establish calibration.",
        },
        "decision_benefit": {
            "passed": reduced["benefit_passed"] if complete else None,
            "measured_operands": reduced["paired_improvements"] if complete else {},
            "principle": "Paired family gains must exceed uncertainty.",
        },
        "retention": {
            "passed": len(all_rows) == len(panel) * 2 if panel else None,
            "measured_operands": {"retained": len(all_rows)},
            "principle": "Invalid and unstarted units remain visible.",
        },
        "efficiency": {
            "passed": None,
            "measured_operands": {
                "input_tokens": sum(row.get("input_tokens", 0) for row in rows),
                "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
                "latency_s": sum(row.get("latency_s", 0) for row in rows),
            },
            "principle": "Token cost does not imply decision benefit.",
        },
    }
    old_path = ROOT / "results/experiment_7770_v676_qwen_runner_qualification.json"
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    historical = {
        "path": str(old_path),
        "sha256": sha256_file(old_path) if old_path.is_file() else None,
        "honest_verdict": old.get("honest_verdict"),
        "qwen_runner_ready_score": old.get("qwen_runner_ready_score"),
        "full_python_suite": next(
            (
                item
                for item in old.get("validation_receipts", {}).get("commands", [])
                if item.get("name") == "full_python_suite"
            ),
            None,
        ),
    }
    model_calls = runtime.get("canary_calls", 0) + runtime.get("panel_calls", 0)
    return {
        "schema": "carnot.exp7787.v677.event_confidence.v1",
        "experiment_id": "exp7787-qwen-event-confidence",
        "milestone": "2026.09.677",
        "run_date": date,
        "honest_verdict": verdict,
        "verdict_class": cls,
        "flagged_adversarial": False,
        "gate_check_summary": failed_checks,
        "rows": all_rows,
        "acceptance_gate_results": gates,
        "duration_s": time.monotonic() - started,
        "phase_spans": spans,
        "random_seed": SEED,
        "reproducibility_checksum": canonical_hash(
            {
                "protocol": protocol,
                "producer_hash": hashes.get("producer"),
                "panel_hash": hashes.get("panel"),
                "code_sha256": sha256_file(Path(__file__)),
                "cli_sha256": sha256_file(
                    ROOT / "scripts/experiments/experiment_7787_v677_qwen_event_confidence.py"
                ),
                "seed": SEED,
            }
        ),
        "sample_size_budget": {
            "intended": 24,
            "eligible": len(panel),
            "started": len({row["family_id"] for row in rows}),
            "completed": reduced["independent_n"],
            "excluded": 0,
            "censored": sum(row.get("censored", False) for row in all_rows),
            "independent_n": reduced["independent_n"],
            "intended_calls": 48,
            "started_calls": len(rows),
            "completed_calls": sum(row.get("disposition") == "completed" for row in rows),
            "unstarted_calls": len(all_rows) - len(rows),
        },
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": receipts,
        "verifier_is_oracle": False,
        "claim_scope": "exposed_natural_development_pilot; no held-out generalization or production activation",
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": "live_llm_inference"
        if model_calls
        else "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "model_bounded_generation" if model_calls else "no_model_load",
        "planned_inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": [MODEL_ID],
        "model_specs": [
            {
                "hf_id": MODEL_ID,
                "model_path": runtime.get("model_path"),
                "model_sha256": runtime.get("model_sha256"),
                "quantization": "Q4_K_M",
            }
        ]
        if runtime.get("model_load_completed")
        else [],
        "model_invocation_counts": {
            "model_loads": runtime.get("model_load_completed", 0),
            "canary_calls": runtime.get("canary_calls", 0),
            "panel_calls": runtime.get("panel_calls", 0),
            "input_tokens": gates["efficiency"]["measured_operands"]["input_tokens"],
            "output_tokens": gates["efficiency"]["measured_operands"]["output_tokens"],
        },
        "qwen_protocol_ready_score": ready,
        "parse_coverage_by_arm": reduced["parse_coverage_by_arm"],
        "semantic_comparison_rows": reduced["semantic_comparison_rows"],
        "paired_improvements": reduced["paired_improvements"],
        "model_file_sha256": runtime.get("model_sha256"),
        "gpu_offload_receipt": runtime.get("gpu_offload_receipt"),
        "historical_full_suite_status": historical,
        "protocol": protocol,
        "panel": panel,
        "reduced": reduced,
        "current_model_receipts": runtime,
        "production_promotion": False,
    }


def span(started: float, phase: str, begin: float, units: int) -> dict[str, Any]:
    """Preserve actual monotonic phase duration without padding."""
    return {
        "phase": phase,
        "start_s": begin - started,
        "end_s": time.monotonic() - started,
        "duration_s": time.monotonic() - begin,
        "completed_units": units,
    }


def run_experiment(root: Path, date: str, output: Path) -> dict[str, Any]:
    """Validate the frozen closure, capture once, read cold, then publish."""
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
    from carnot.inference.sota_models import cached_current_model

    started = time.monotonic()
    spans: list[dict[str, Any]] = []
    raw_dir = root / RAW
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress(started, "preconditions", "before")
    begin = time.monotonic()
    panel, checks, hashes = preflight(root)
    context: dict[str, Any] = {}
    if panel:
        reconstructed, extra_checks, _, context = source.prepare_panel(root, started)
        checks.extend(extra_checks)
        checks.append(
            gate(
                "frozen_panel_bytes",
                "exp7745_pre_gate",
                root / PANEL,
                "reconstructed_equals_frozen",
                True,
                reconstructed == panel,
            )
        )
        model = cached_current_model()
        checks.append(
            gate(
                "required_model",
                "local_cache",
                Path(str((model or {}).get("model_path") or "/missing.gguf")),
                "hf_id",
                MODEL_ID,
                (model or {}).get("hf_id"),
            )
        )
        checks.append(
            gate(
                "required_model_path",
                "local_cache",
                Path(str((model or {}).get("model_path") or "/missing.gguf")),
                "model_path",
                str(context.get("model_path")),
                str((model or {}).get("model_path")),
            )
        )
        old = json.loads((root / PRODUCER).read_text())
        checks.append(
            gate(
                "model_file_hash",
                "local_cache",
                Path(str(context.get("model_path") or "/missing.gguf")),
                "sha256",
                old.get("current_model_receipts", {}).get("model_sha256"),
                context.get("model_sha256"),
            )
        )
        if any(not check["passed"] for check in checks):
            panel = []
    spans.append(span(started, "preconditions", begin, len(checks)))
    progress(started, "preconditions", "after", len(checks))
    protocol = make_protocol(panel) if panel else None
    if protocol:
        atomic_json(raw_dir / "protocol.json", protocol)
        atomic_json(raw_dir / "frozen_panel.json", panel)
    receipts: dict[str, Any] = {
        "frozen_scope": json.loads(Path("/tmp/exp7787/frozen_scope.json").read_text()),
        "required": [],
        "repository_health": [],
    }
    runtime: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if panel and not any(not check["passed"] for check in checks):
        tests = (
            receipts["frozen_scope"]["direct_tests"]
            + receipts["frozen_scope"]["imported_consumer_closure"]
        )
        commands = []
        for index, test in enumerate(tests):
            basetemp = Path(f"/tmp/exp7787/basetemp/preflight_{index}")
            basetemp.mkdir(parents=True, exist_ok=True)
            commands.append(
                CommandSpec(
                    f"affected_pytest_{index}",
                    (
                        str(root / ".venv/bin/pytest"),
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        f"--basetemp={basetemp}",
                        test,
                        "-q",
                    ),
                    "frozen_affected_tests",
                    300,
                )
            )
        begin = time.monotonic()
        progress(started, "affected_validation", "before")
        receipts["required"] += run_commands(
            root, commands, log_dir=raw_dir / "validation" / "pre_capture", heartbeat_s=45
        )
        spans.append(span(started, "affected_validation", begin, len(commands)))
        progress(started, "affected_validation", "after", len(commands))
        if all(item["passed"] for item in receipts["required"]):
            begin = time.monotonic()
            progress(started, "capture", "before")
            try:
                rows, runtime = owned_capture(root, panel, protocol, context, started)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                runtime["capture_error"] = f"{type(error).__name__}:{error}"
            spans.append(span(started, "capture", begin, len(rows)))
            progress(started, "capture", "after", len(rows))
    if panel:
        tests = receipts["frozen_scope"]["direct_tests"]
        base = Path("/tmp/exp7787/basetemp/post")
        base.mkdir(parents=True, exist_ok=True)
        cov = Path("/tmp/exp7787/coverage")
        module = "python/carnot/experiment_7787_v677_qwen_event_confidence.py"
        cli = "scripts/experiments/experiment_7787_v677_qwen_event_confidence.py"
        commands = [
            CommandSpec(
                "new_module_coverage",
                (
                    str(root / ".venv/bin/coverage"),
                    "run",
                    f"--data-file={cov}",
                    "--include=*/experiment_7787_v677_qwen_event_confidence.py",
                    "-m",
                    "pytest",
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    f"--basetemp={base / 'coverage'}",
                    *tests,
                    "-q",
                ),
                "new_code",
                300,
            ),
            CommandSpec(
                "new_module_coverage_report",
                (
                    str(root / ".venv/bin/coverage"),
                    "report",
                    f"--data-file={cov}",
                    "--include=*/experiment_7787_v677_qwen_event_confidence.py",
                    "--show-missing",
                    "--fail-under=100",
                ),
                "new_code",
                60,
            ),
            CommandSpec(
                "ruff_check",
                (str(root / ".venv/bin/ruff"), "check", module, cli, *tests),
                "changed_files",
                60,
            ),
            CommandSpec(
                "ruff_format",
                (str(root / ".venv/bin/ruff"), "format", "--check", module, cli, *tests),
                "changed_files",
                60,
            ),
            CommandSpec("mypy", (str(root / ".venv/bin/mypy"), module), "changed_module", 180),
            CommandSpec(
                "spec_coverage",
                (str(root / ".venv/bin/python"), "scripts/check_spec_coverage.py", *tests),
                "explicit_tests",
                180,
            ),
        ]
        (base / "coverage").mkdir(parents=True, exist_ok=True)
        begin = time.monotonic()
        progress(started, "scoped_validation", "before")
        receipts["required"] += run_commands(
            root, commands, log_dir=raw_dir / "validation" / "scoped", heartbeat_s=45
        )
        spans.append(span(started, "scoped_validation", begin, len(commands)))
        progress(started, "scoped_validation", "after", len(commands))
    artifact = build_artifact(
        panel, protocol, rows, checks, hashes, runtime, receipts, started, spans, date
    )
    candidate = raw_dir / "candidate.json"
    atomic_json(candidate, artifact)
    readers = [
        CommandSpec(
            "cold_replay",
            (
                str(root / ".venv/bin/python"),
                "-u",
                "scripts/experiments/experiment_7787_v677_qwen_event_confidence.py",
                "--cold-replay",
                str(candidate),
            ),
            "exact_candidate",
            120,
        ),
        CommandSpec(
            "adversarial_verify",
            (
                str(root / ".venv/bin/python"),
                "scripts/adversarial_verify.py",
                str(candidate),
                "--json",
            ),
            "exact_candidate",
            120,
        ),
        CommandSpec(
            "strict_row_consistency",
            (
                str(root / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_candidate",
            120,
        ),
    ]
    begin = time.monotonic()
    progress(started, "terminal_readers", "before")
    receipts["required"] += run_commands(
        root, readers, log_dir=raw_dir / "validation" / "terminal", heartbeat_s=45
    )
    spans.append(span(started, "terminal_readers", begin, len(readers)))
    progress(started, "terminal_readers", "after", len(readers))
    artifact = build_artifact(
        panel, protocol, rows, checks, hashes, runtime, receipts, started, spans, date
    )
    artifact["flagged_adversarial"] = not next(
        item["passed"] for item in receipts["required"] if item["name"] == "adversarial_verify"
    )
    artifact["validation_receipts"]["terminal_candidate_sha256"] = sha256_file(candidate)
    atomic_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Expose one dated capture command and an independent cold reader."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260927")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        print(json.dumps(cold_reduce(args.cold_replay), sort_keys=True), flush=True)
        return 0
    if args.date != "20260927":
        parser.error("run_date_mismatch")
    artifact = run_experiment(ROOT, args.date, ROOT / RESULT)
    print(
        json.dumps({"honest_verdict": artifact["honest_verdict"], "result": str(RESULT)}),
        flush=True,
    )
    return 0
