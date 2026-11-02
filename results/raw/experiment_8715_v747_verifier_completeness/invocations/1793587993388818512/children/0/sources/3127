"""Run the fixed paired Qwen record proposal diagnostic. REQ-REPORT-7702."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7665_v668_qwen_grounded_claims as launch
from carnot import experiment_7676_v669_qwen_quote_relations as prior
from carnot import experiment_7700_v671_record_span_protocol as protocol
from carnot.reporting import current_work_receipt as custody
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.verify.record_addresses import analyze_answer
from carnot.verify.record_proposals import ARMS, reduce_response
from carnot.verify.tool_source_atoms import digest


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7702_v671_qwen_record_pilot")
RESULT = Path("results/experiment_7702_v671_qwen_record_pilot.json")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
SEED = 7702
MODULE = "python/carnot/experiment_7702_v671_qwen_record_pilot.py"
CAPABILITY = "python/carnot/verify/record_proposals.py"
TEST = "tests/python/test_experiment_7702_v671_qwen_record_pilot.py"
WRAPPER = "scripts/experiments/experiment_7702_v671_qwen_record_pilot.py"
SCOPE = {
    "tests": [TEST],
    "changed_modules": [MODULE, CAPABILITY],
    "static_paths": [WRAPPER],
    "specs": ["REQ-REPORT-7702", "REQ-VERIFY-7702"],
    "e2e": ["task_cold_raw_replay"],
}


def progress(started: float, phase: str, event: str, **detail: Any) -> None:  # pragma: no cover
    """Flush each phase and long model wait with elapsed time and units."""

    suffix = " ".join(f"{key}={value}" for key, value in detail.items())
    print(
        f"[exp7702] {phase} {event} elapsed_s={time.monotonic() - started:.2f} {suffix}",
        flush=True,
    )


def gate(
    check: str, upstream_id: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep exact operands for every missing or false prerequisite."""

    return {
        "check": check,
        "upstream_id": upstream_id,
        "artifact_path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def freeze_panel(pilots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse the exact sixteen Exp7676 fixtures and eight authenticated pilots."""

    panel = prior.freeze_panel(pilots)
    for row in panel:
        row["analysis"] = analyze_answer(row["source"], row["answer"])
        row["source_sha256"] = digest(row["source"])
        row["answer_sha256"] = digest(row["answer"])
    return panel


def make_request(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """Expose original bytes in both arms, with labels omitted from record tables."""

    if arm not in ARMS:
        raise ValueError("unplanned_arm")
    base = {"complete_source": row["source"], "complete_answer": row["answer"]}
    if arm == "opaque_index":
        instruction = (
            "/no_think\nReturn one compact JSON object with proposal: claim_index (zero based), "
            "kind, arguments, polarity, modifiers, relation, and source_quote. "
            "Copy an exact source quote. Use relation unknown when unsure."
        )
    else:
        analysis = row["analysis"]
        base["records"] = [
            {
                "record_id": record["record_id"],
                "source_bytes": record["source_bytes"],
                "byte_start": record["byte_start"],
                "byte_end": record["byte_end"],
                "dialect": record["dialect"],
                "source_id": record["source_id"],
                "line": record["line"],
                "arguments": [item["arguments"] for item in record["tuples"]],
            }
            for record in analysis["records"]
        ]
        base["sentences"] = [
            {"sentence_id": item["sentence_id"], "full_text": item["full_text"]}
            for item in analysis["sentences"]
        ]
        base["propositions"] = [
            {
                "proposition_id": item["proposition_id"],
                "sentence_id": item["sentence_id"],
                "text": item["text"],
            }
            for item in analysis["propositions"]
        ]
        instruction = (
            "/no_think\nReturn one compact JSON object with proposal: sentence_id, "
            "proposition_id, kind, arguments, polarity, modifiers, relation, and source_quote. "
            "Use the full answer sentence and one exact quoted substring from a source record. "
            "A source location alone does not prove a claim. Use unknown when unsure."
        )
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(base, ensure_ascii=False)},
        ],
        "temperature": 0,
        "seed": SEED,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


def preflight(
    root: Path, started: float
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate current milestone schema, model bytes, offload and owned GPU."""

    from carnot import experiment_7630_v666_cuda_ownership as ownership
    from carnot.inference.sota_models import cached_current_model
    from llama_cpp import llama_cpp

    root = root.resolve()
    checks = [
        gate(
            "absolute_root",
            "current_work",
            str(root),
            "absolute_directory",
            True,
            root.is_absolute() and root.is_dir(),
        )
    ]
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_custody": []}
    for relative, upstream in (
        (protocol.RESULT, "exp7700"),
        (protocol.PILOT, "exp7602_input_bytes"),
        (Path("ops/exclusion_manifest.yaml"), "operator_exclusion_manifest"),
    ):
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate("required_input", upstream, str(path), "readable_nonempty_bytes", True, present)
        )
        if present:
            hashes["producers"][str(relative)] = custody.sha256_file(path)
        else:
            hashes["missing_custody"].append(str(relative))
    if any(not check["passed"] for check in checks):
        return checks, hashes, {}
    qualified = json.loads((root / protocol.RESULT).read_text(encoding="utf-8"))
    for field, expected in (
        ("record_protocol_ready_score", 1),
        ("verdict_class", "circular_positive"),
        ("flagged_adversarial", False),
        ("feature_schema_path", str(protocol.RAW / "feature_schema.json")),
    ):
        checks.append(
            gate(
                "exp7700_qualified",
                "exp7700",
                str(root / protocol.RESULT),
                field,
                expected,
                qualified.get(field),
            )
        )
    schema_path = root / str(qualified.get("feature_schema_path", "missing"))
    checks.append(
        gate(
            "exp7700_schema",
            "exp7700",
            str(schema_path),
            "readable_nonempty_bytes",
            True,
            schema_path.is_file() and schema_path.stat().st_size > 0,
        )
    )
    if schema_path.is_file():
        hashes["producers"][str(schema_path.relative_to(root))] = custody.sha256_file(schema_path)
    if any(not check["passed"] for check in checks):
        return checks, hashes, {}
    model = cached_current_model(preferred_quant="Q4_K_M")
    model_path = Path(str((model or {}).get("model_path") or "/missing-qwen.gguf"))
    model_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and "Q4_K_M" in model_path.name
        and model_path.is_file()
        and model_path.stat().st_size > 15_000_000_000
    )
    checks.append(
        gate(
            "cached_qwen_identity",
            "local_model_cache",
            str(model_path),
            "hf_id_quantization_bytes",
            True,
            model_ok,
        )
    )
    offload = bool(llama_cpp.llama_supports_gpu_offload())
    checks.append(
        gate(
            "cuda_offload_support",
            "llama_cpp_build",
            str(model_path),
            "llama_supports_gpu_offload",
            True,
            offload,
        )
    )
    if not model_ok or not offload:
        return checks, hashes, {}
    progress(started, "model_hash", "before")
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
    return (
        checks,
        hashes,
        {
            "model": model,
            "model_path": model_path,
            "model_sha256": model_hash,
            "registry": registry,
            "selected": selected,
            "inventory": inventory,
            "ownership_rows": ownership_rows,
        },
    )


def build_artifact(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    hashes: dict[str, Any],
    runtime: dict[str, Any],
    spans: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    """Keep terminal task accounting separate from oracle-bound scientific claims."""

    blocked = bool(failures and not runtime.get("model_load_attempted"))
    partial = not blocked and (len(rows) < 48 or bool(failures))
    false_support = sum(
        row["metrics"]["false_support"] for row in rows if row["population"] == "fixture"
    )
    valid = not failures
    complete = len(rows) == 48 and all(row.get("raw_response_sha256") for row in rows)
    ready = bool(valid and complete and false_support == 0)
    verdict_class = (
        "blocked" if blocked else "partial" if partial else "circular_positive" if ready else "null"
    )
    verdict = (
        f"complete_blocked_{failures[0]['check']}"
        if blocked
        else "complete_partial_owned_generation"
        if partial
        else "complete_circular_positive_fixture_protocol"
        if ready
        else "complete_null_record_proposal_protocol"
    )
    groups = {row["unit_id"] for row in rows}
    gate_values = {
        "validity": (valid, {"failed_checks": len(failures)}),
        "readiness": (ready, {"completed_calls": len(rows), "false_support": false_support}),
        "coverage": (complete, {"independent_groups": len(groups), "paired_calls": len(rows)}),
        "freshness": (None, {"fresh_natural_groups": 0, "exposed_pilots": 8}),
        "probability": (None, {"independent_probability_labels": 0}),
        "utility": (None, {"measured_decision_outcomes": 0}),
        "retention": (None, {"delayed_replay_groups": 0}),
        "efficiency": (
            None,
            {
                "input_tokens": sum(row.get("prompt_tokens", 0) for row in rows),
                "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
                "duration_s": duration,
            },
        ),
    }
    gate_principles = {
        "validity": "Failed checks prevent invalid evidence propagation.",
        "readiness": "Zero false support is required for fixture protocol qualification.",
        "coverage": "Quality thresholds prevent effects being inferred from plumbing.",
        "freshness": "Exposed pilots cannot establish fresh accuracy.",
        "probability": "A probability claim needs independent labels.",
        "utility": "Utility requires measured outcomes.",
        "retention": "Retention bounds prevent improvement by forgetting.",
        "efficiency": "Efficiency requires a comparable measured workload.",
    }
    gates = {
        key: {"passed": passed, "measured_operands": operands, "principle": gate_principles[key]}
        for key, (passed, operands) in gate_values.items()
    }
    principles = {
        name: "This record makes the stated claim independently checkable and limits its scope."
        for name in (
            "honest_verdict",
            "verdict_class",
            "flagged_adversarial",
            "gate_check_summary",
            "acceptance_gate_results",
            "rows",
            "sample_size_budget",
            "inference_substrate",
            "inference_substrate_class",
            "MODEL_SPECS",
            "model_invoked",
            "execution_venue",
            "phase_spans",
            "random_seed",
            "source_artifact_hashes",
            "preconditions_checked",
            "validation_receipts",
            "verifier_is_oracle",
            "field_principles",
            "qwen_pilot_complete_score",
            "current_model_receipts",
            "addressing_metrics",
        )
    }
    principles.update({f"acceptance_gate_{key}": value for key, value in gate_principles.items()})
    paired = []
    for unit_id in sorted(groups):
        arms = {row["arm"]: row for row in rows if row["unit_id"] == unit_id}
        if len(arms) == 2:
            old, new = arms["opaque_index"], arms["explicit_record"]
            paired.append(
                {
                    "unit_id": unit_id,
                    "schema_valid_delta": int(new["metrics"]["schema_valid"])
                    - int(old["metrics"]["schema_valid"]),
                    "located_delta": int(new["metrics"]["unique_containing_record"])
                    - int(old["metrics"]["unique_containing_record"]),
                    "binding_delta": int(new["metrics"]["correct_binding"])
                    - int(old["metrics"]["correct_binding"]),
                    "support_delta": int(new["metrics"]["supported"])
                    - int(old["metrics"]["supported"]),
                    "censored": old["censored"] or new["censored"],
                }
            )
    invoked = bool(runtime.get("model_load_attempted"))
    backend = "codex" if os.environ.get("CODEX_SESSION_ID") else "unknown"
    artifact = {
        "schema": "carnot.exp7702.v671.qwen_record_pilot.v1",
        "experiment_id": "exp7702-qwen-record-pilot",
        "milestone": "2026.09.671",
        "run_date": "20260926",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "acceptance_gate_results": gates,
        "rows": rows,
        "paired_group_differences": paired,
        "sample_size_budget": {
            "intended": 24,
            "observed": len(groups),
            "eligible": len(groups),
            "excluded": 0,
            "censored": len({row["unit_id"] for row in rows if row["censored"]}),
            "effective_blocks": len(paired),
            "prior_exposure": "Eight V664/V669 exposed pilots and sixteen exact fixture oracles.",
            "inference_limits": "48 calls; one proposal each; 256 output tokens per call; no fresh accuracy claim.",
        },
        "inference_substrate": "owned_local_llama_cpp_paired_bounded_generation"
        if invoked
        else "no_model_load",
        "inference_substrate_class": "model_bounded_generation" if invoked else "no_model_load",
        "planned_inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": [MODEL_ID] if invoked else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_specs": [
            {"hf_id": MODEL_ID, "model_path": runtime.get("model_path"), "quantization": "Q4_K_M"}
        ]
        if invoked
        else [{"model": "none", "reason": "blocked_before_model_load"}],
        "model_invoked": invoked,
        "invocation_counts": {
            "loads": runtime.get("model_load_attempted", 0),
            "forwards": runtime.get("generation_attempted", 0),
            "generations": len(rows),
            "input_tokens": sum(row.get("prompt_tokens", 0) for row in rows),
            "output_tokens": sum(row.get("output_tokens", 0) for row in rows),
            "failures": len(failures),
            "cancellations": max(0, runtime.get("generation_attempted", 0) - len(rows)),
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "host_pid": os.getpid(),
            "server_pid": runtime.get("server_pid"),
            "gpu_uuid": runtime.get("device_uuid"),
        },
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {
            "paired_generation": SEED,
            "fixture_selection": "stack:0-5,grep:0-4,ast:0-4",
        },
        "reproducibility_checksum": custody.canonical_hash(
            {
                "producers": hashes.get("producers", {}),
                "seed": SEED,
                "arms": ARMS,
                "max_tokens": 256,
                "reducer_sha256": custody.sha256_file(ROOT / CAPABILITY),
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
        "verifier_is_oracle": True,
        "field_principles": principles,
        "qwen_pilot_complete_score": int(complete and valid),
        "current_model_receipts": {
            **runtime,
            "requests": [
                {
                    "path": row["request_path"],
                    "sha256": row["request_sha256"],
                    "response_path": row["raw_response_path"],
                    "response_sha256": row["raw_response_sha256"],
                }
                for row in rows
            ],
        },
        "addressing_metrics": [
            {
                "unit_id": row["unit_id"],
                "arm": row["arm"],
                **{
                    key: row["metrics"][key]
                    for key in (
                        "schema_valid",
                        "exact_span",
                        "unique_containing_record",
                        "claim_alignment",
                        "correct_binding",
                        "tuple_truth",
                        "unknown_remainder",
                        "truncated",
                    )
                },
            }
            for row in rows
        ],
        "execution_recovery_receipt": {
            "declared_backend": "codex",
            "effective_backend": backend,
            "force_experiments": os.environ.get("CODEX_FORCE_EXPERIMENTS"),
            "successful_invocation": {
                "kind": "current_api_assistant_invocation",
                "backend": backend,
                "session_id": os.environ.get("CODEX_SESSION_ID"),
            },
            "future_quota_guaranteed": False,
        },
        "prior_failures": [
            {
                "experiment_id": "exp7688",
                "custody": "not_emitted_usage_limit_three_attempts",
                "scientific_verdict": None,
            }
        ],
        "activation": False,
        "production_promotion": False,
    }
    return artifact


def independent_reduce(path: Path) -> dict[str, Any]:
    """Cold replay exact request and response bytes with a newly imported reader."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact["rows"]
    if artifact["verdict_class"] == "blocked":
        return {"passed": not rows and not artifact["model_invoked"], "calls": 0}
    for row in rows:
        request_path = Path(row["request_path"])
        response_path = Path(row["raw_response_path"])
        if (
            custody.sha256_file(request_path) != row["request_sha256"]
            or custody.sha256_file(response_path) != row["raw_response_sha256"]
        ):
            return {"passed": False, "reason": "raw_hash_mismatch"}
        request = json.loads(request_path.read_text(encoding="utf-8"))
        visible = json.loads(request["messages"][1]["content"])
        if (
            visible["complete_source"] != row["source"]
            or visible["complete_answer"] != row["answer"]
        ):
            return {"passed": False, "reason": "request_source_answer_mismatch"}
        response = json.loads(response_path.read_text(encoding="utf-8"))
        choice = response["choices"][0]
        content = str(choice["message"].get("content") or "")
        finish = str(choice.get("finish_reason") or "unknown")
        if content != row["response_text"] or finish != row["finish_reason"]:
            return {"passed": False, "reason": "raw_response_mismatch"}
        if reduce_response(row, row["arm"], content, finish) != row["metrics"]:
            return {"passed": False, "reason": "reduction_mismatch"}
    return {"passed": len(rows) == 48, "calls": len(rows)}


def cold_replay(path: Path) -> dict[str, Any]:
    """Recompute both arms and fixed panel identities after process restart."""

    reduced = independent_reduce(path)
    if not reduced["passed"]:
        return reduced
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["verdict_class"] == "blocked":
        return reduced
    panel = json.loads((ROOT / RAW / "frozen_panel.json").read_text(encoding="utf-8"))
    saved = {(row["unit_id"], row["arm"]) for row in artifact["rows"]}
    expected = {(row["unit_id"], arm) for row in panel for arm in ARMS}
    return {"passed": len(panel) == 24 and saved == expected, "calls": len(saved)}


def terminal_commands(root: Path, candidate: Path) -> list[validation.CommandSpec]:
    """Run cold readers and existing strict terminal gates on one candidate."""

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


def run_experiment(root: Path, date: str, output: Path) -> int:  # pragma: no cover
    """Checkpoint bounded owned work, validate, then publish a terminal record."""

    started = time.monotonic()
    root = root.resolve()
    progress(started, "startup", "before", root=root)
    if date != "20260926":
        raise SystemExit("--date must be 20260926")
    destination = output if output.is_absolute() else root / output
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7702-"))
    (root / RAW).mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, context = preflight(root, started)
    failures = [row for row in checks if not row["passed"]]
    spans.append(
        launch._span("preconditions", started, begin, len(checks), len(checks) - len(failures))
    )
    progress(started, "preconditions", "after", failures=len(failures))
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    if not failures:
        begin = time.monotonic()
        progress(started, "freeze_panel", "before")
        try:
            pilots = [
                json.loads(line)
                for line in (root / protocol.PILOT).read_text(encoding="utf-8").splitlines()
            ]
            panel = freeze_panel(pilots)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            failure = gate(
                "pilot_authentication",
                "exp7602",
                str(root / protocol.PILOT),
                "source_answer_hashes",
                True,
                f"{type(error).__name__}:{error}",
            )
            checks.append(failure)
            failures.append(failure)
        else:
            custody.atomic_json(root / RAW / "frozen_panel.json", panel)
            custody.atomic_json(root / RAW / "frozen_affected_validation_manifest.json", SCOPE)
            hashes["pre_gate_receipts"]["frozen_panel"] = custody.sha256_file(
                root / RAW / "frozen_panel.json"
            )
        spans.append(
            launch._span("freeze_panel", started, begin, 24, 0 if failures else len(panel))
        )
        progress(started, "freeze_panel", "after", completed_units=0 if failures else len(panel))
    if not failures:
        begin = time.monotonic()
        progress(started, "measurement", "before", planned_calls=48)
        try:
            rows, runtime = launch._measure(
                root,
                panel,
                context,
                started,
                arms=ARMS,
                request_builder=make_request,
                response_reducer=reduce_response,
                raw_path=RAW,
                task_id="experiment_7702_v671_qwen_record_pilot",
            )
            runtime["model_path"] = str(context["model_path"])
            runtime["model_sha256"] = context["model_sha256"]
        except BaseException as error:
            run_dirs = sorted((root / RAW / "runs").glob(f"*-{os.getpid()}"))
            checkpoint = run_dirs[-1] / "checkpoint.json" if run_dirs else None
            if checkpoint and checkpoint.is_file():
                rows = json.loads(checkpoint.read_text(encoding="utf-8"))["rows"]
            runtime = {
                "model_load_attempted": 1,
                "generation_attempted": len(rows) + 1,
                "model_path": str(context["model_path"]),
                "model_sha256": context["model_sha256"],
                "device_uuid": context["selected"]["uuid"],
            }
            failure = gate(
                "owned_generation",
                "exp7630_owned_qwen_server",
                str(root / RAW),
                "forty_eight_completed_calls",
                48,
                f"{len(rows)}:{type(error).__name__}:{error}",
            )
            checks.append(failure)
            failures.append(failure)
        spans.append(launch._span("measurement", started, begin, 48, len(rows)))
        progress(started, "measurement", "after", completed_units=len(rows), failures=len(failures))
    rows_path = root / RAW / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    hashes["pre_gate_receipts"]["raw_rows"] = custody.sha256_file(rows_path)
    artifact = build_artifact(rows, failures, hashes, runtime, spans, time.monotonic() - started)
    artifact["preconditions_checked"] = checks
    begin = time.monotonic()
    progress(started, "scoped_validation", "before")
    basetemp = private / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands = validation.build_scoped_commands(
        root,
        SCOPE["tests"],
        SCOPE["changed_modules"],
        static_paths=SCOPE["static_paths"],
        basetemp=basetemp,
        coverage_file=private / ".coverage.exp7702",
    )
    receipts = validation.run_commands(
        root, commands, log_dir=private / "logs" / "scoped", heartbeat_s=45
    )
    spans.append(launch._span("scoped_validation", started, begin, len(commands), len(receipts)))
    progress(
        started,
        "scoped_validation",
        "after",
        completed_units=len(receipts),
        passed=all(row["passed"] for row in receipts),
    )
    artifact["validation_receipts"]["required_commands"] = receipts
    if not all(row["passed"] for row in receipts):
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    candidate = private / "terminal_candidate.json"
    custody.atomic_json(candidate, artifact)
    begin = time.monotonic()
    progress(started, "terminal_readers", "before", candidate=candidate)
    terminal = validation.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=private / "logs" / "terminal",
        heartbeat_s=45,
    )
    spans.append(launch._span("terminal_readers", started, begin, 4, len(terminal)))
    progress(
        started,
        "terminal_readers",
        "after",
        completed_units=len(terminal),
        passed=all(row["passed"] for row in terminal),
    )
    artifact["validation_receipts"]["terminal_readers"] = terminal
    artifact["flagged_adversarial"] = not next(
        row["passed"] for row in terminal if row["name"] == "adversarial_verify"
    )
    if not all(row["passed"] for row in receipts + terminal):
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
        artifact["qwen_pilot_complete_score"] = 0
        for result in artifact["acceptance_gate_results"].values():
            result["passed"] = False
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    custody.atomic_json(destination, artifact)
    progress(
        started,
        "publish",
        "after",
        path=destination,
        verdict=artifact["honest_verdict"],
        sha256=custody.sha256_file(destination),
    )
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Accept the frozen run date and two cold-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260926")
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
