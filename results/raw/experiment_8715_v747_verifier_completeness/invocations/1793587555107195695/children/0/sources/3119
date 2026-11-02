"""Run paired Qwen proposals for exact quote and numeric source addressing.

This diagnostic compares independently checked relation content. Exposed pilots
and fixture oracles cannot establish broad answer accuracy. REQ-REPORT-7676.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7665_v668_qwen_grounded_claims as previous
from carnot import experiment_7672_v669_bound_relations as fixtures
from carnot.reporting import current_work_receipt as custody
from carnot.reporting import experiment_7303_validation_scope as validation
from carnot.verify.quote_relations import ARMS, reduce_proposal
from carnot.verify.tool_source_atoms import digest
from carnot.verify.tool_source_relations import verify_relations


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7676_v669_qwen_quote_relations")
RESULT = Path("results/experiment_7676_v669_qwen_quote_relations.json")
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
SEED = 7676
MODULE = "python/carnot/experiment_7676_v669_qwen_quote_relations.py"
CAPABILITY = "python/carnot/verify/quote_relations.py"
TEST = "tests/python/test_experiment_7676_v669_qwen_quote_relations.py"
WRAPPER = "scripts/experiments/experiment_7676_v669_qwen_quote_relations.py"
MANIFEST = {
    "tests": [TEST],
    "changed_modules": [MODULE, CAPABILITY],
    "static_paths": [WRAPPER, previous.MODULE],
    "specs": ["REQ-REPORT-7676", "REQ-VERIFY-7676"],
}


def freeze_panel(pilots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Authenticate the exposed inputs before attaching any narrow truth."""
    if len(pilots) != 8 or len({row["component_hash"] for row in pilots}) != 8:
        raise ValueError("eight_distinct_pilots_required")
    selected = [
        case
        for case in fixtures.fixture_cases()
        if (case["dialect"] == "stack" and int(case["id"][-2:]) < 6)
        or (case["dialect"] in {"grep", "ast"} and int(case["id"][-2:]) < 5)
    ]
    if len(selected) != 16:
        raise ValueError("sixteen_relation_fixtures_required")
    panel = []
    for pilot in pilots:
        source, answer = pilot["complete_source"], pilot["complete_answer"]
        if pilot["source_sha256"] != digest(source) or pilot["answer_sha256"] != digest(answer):
            raise ValueError("pilot_input_authentication_failure")
        panel.append(
            {
                "unit_id": pilot["component_hash"],
                "population": "pilot",
                "source": source,
                "answer": answer,
                "fixture_truth": None,
                "prior_exposure": True,
            }
        )
    for case in selected:
        panel.append(
            {
                "unit_id": case["id"],
                "population": "fixture",
                "source": case["source"],
                "answer": case["answer"],
                "fixture_truth": case["truth"],
                "prior_exposure": True,
                "attack": case["attack"],
            }
        )
    for row in panel:
        row["source_sha256"] = digest(row["source"])
        row["answer_sha256"] = digest(row["answer"])
        row["narrow_truth"] = verify_relations(row["source"], row["answer"])
        row["truth"] = row["narrow_truth"]
    return panel


def make_request(row: dict[str, Any], arm: str) -> dict[str, Any]:
    """Keep source and answer bytes identical across the two proposal arms."""
    if arm not in ARMS:
        raise ValueError("invalid_arm")
    address = (
        "source_byte_start and source_byte_end as UTF-8 byte integers"
        if arm == "numeric_offset"
        else "source_quote copied exactly from source"
    )
    instruction = (
        "/no_think\nReturn one compact JSON object with proposals array. For each proposal use "
        "claim_index (zero based), kind (stack_frame, grep_quote, definition_in_scope), "
        "arguments (all named tuple fields), polarity (positive or negative), modifiers "
        "(all causal, modal, quantifier and compound words), relation "
        "(supports, contradicts, unknown), and " + address + ". "
        "Preserve every qualifier in the original answer. Use [] when uncertain."
    )
    visible = {"complete_source": row["source"], "complete_answer": row["answer"]}
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


def _gate(
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> dict[str, Any]:
    return previous.gate(check, upstream, path, field, expected, observed)


def _acceptance(rows: list[dict[str, Any]], valid: bool) -> dict[str, Any]:
    complete = len(rows) == 48 and all(not row["censored"] for row in rows)
    supported = sum(row["metrics"]["full_proposition_supported"] for row in rows)
    names = {
        "validity": (valid, {"required_checks_passed": valid}),
        "readiness": (
            valid and complete and supported > 0,
            {"completed_calls": len(rows), "supported_relations": supported},
        ),
        "coverage": (
            valid and supported > 0,
            {
                "independent_groups": len({row["unit_id"] for row in rows}),
                "supported_relations": supported,
            },
        ),
        "freshness": (False, {"fresh_confirmatory_groups": 0}),
        "probability": (None, {"independent_probability_labels": 0}),
        "decision_utility": (None, {"decision_outcomes": 0}),
        "retention": (None, {"restarts": 0}),
        "efficiency": (
            None,
            {
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "generation_s": sum(row["generation_s"] for row in rows),
            },
        ),
    }
    return {
        key: {
            "passed": passed,
            "measured_operands": operands,
            "principle": "Only measured independent content can open a gate.",
        }
        for key, (passed, operands) in names.items()
    }


def build_artifact(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    hashes: dict[str, Any],
    runtime: dict[str, Any],
    spans: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    """Distinguish completed accounting from independently useful content."""
    blocked = bool(failures and not runtime.get("model_load_attempted"))
    partial = bool(not blocked and (len(rows) < 48 or failures))
    verdict_class = "blocked" if blocked else "partial" if partial else "null"
    verdict = (
        f"complete_blocked_{failures[0]['check']}"
        if blocked
        else "complete_partial_owned_generation"
        if partial
        else "complete_null_quote_relation_diagnostic"
    )
    groups = {row["unit_id"] for row in rows}
    counts = {
        key: sum(row["metrics"].get(key, 0) for row in rows)
        for key in (
            "unique_evidence",
            "full_proposition_supported",
            "contradicted_content",
            "unknown_content",
            "qualifier_retained",
            "unknown_remainder",
        )
    }
    counts["schema_valid"] = sum(bool(row["metrics"].get("schema_valid")) for row in rows)
    artifact = {
        "schema": "carnot.exp7676.v669.qwen_quote_relations.v1",
        "experiment_id": "experiment_7676_v669_qwen_quote_relations",
        "milestone": "2026.09.669",
        "run_date": "20260926",
        "status": "complete",
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "gate_check_summary": {
            "failed_checks": [failure["check"] for failure in failures],
            "failed_count": len(failures),
            "first_failure": failures[0] if failures else None,
        },
        "rows": rows,
        "quote_grounding_rows_path": str(RAW / "rows.jsonl"),
        "proposal_metrics": counts,
        "sample_size_budget": {
            "independent_unit": "source_group",
            "intended": 24,
            "observed": len(groups),
            "eligible": len(groups),
            "excluded": 0,
            "censored": len({row["unit_id"] for row in rows if row["censored"]}),
            "prior_exposure": "Eight real exposed pilots and sixteen exact fixture oracles.",
            "effective_blocks": len(groups),
            "limits": "Diagnostic only; no broad accuracy or fresh decision score.",
        },
        "inference_substrate": (
            "owned_local_llama_cpp_paired_bounded_generation"
            if runtime.get("model_load_attempted")
            else "no_model_load"
        ),
        "inference_substrate_class": (
            "model_bounded_generation" if runtime.get("model_load_attempted") else "no_model_load"
        ),
        "planned_inference_substrate_class": "model_bounded_generation",
        "MODEL_SPECS": [MODEL_ID] if runtime.get("model_load_attempted") else [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "model_specs": (
            [{"hf_id": MODEL_ID, "model_path": runtime.get("model_path"), "quantization": "Q4_K_M"}]
            if runtime.get("model_load_attempted")
            else [{"model": "none", "reason": "blocked_before_model_load"}]
        ),
        "model_invoked": bool(runtime.get("model_load_attempted")),
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
        "execution_venue": "host",
        "execution_venue_details": {
            "host_pid": os.getpid(),
            "gpu_uuid": runtime.get("device_uuid"),
            "owned_server_pid": runtime.get("server_pid"),
        },
        "phase_spans": spans,
        "duration_s": duration,
        "random_seed": {
            "paired_generation": SEED,
            "fixture_selection": "stack:0-5,grep:0-4,ast:0-4",
        },
        "reproducibility_checksum": custody.canonical_hash(
            {
                "input_hashes": hashes.get("producers", {}),
                "seed": SEED,
                "arms": ARMS,
                "max_tokens": 256,
                "reducer_sha256": custody.sha256_file(ROOT / CAPABILITY),
            }
        ),
        "source_artifact_hashes": hashes,
        "preconditions_checked": [],
        "validation_receipts": [],
        "current_model_receipts": runtime,
        "verifier_is_oracle": True,
        "fixture_and_real_dispositions": {
            "fixture": "circular_positive_only",
            "pilot": "exposed_diagnostic_unknown_truth",
        },
        "quote_pilot_measurement_complete_score": int(
            len(rows) == 48 and all(row.get("raw_response_sha256") for row in rows)
        ),
        "retirement_decision": {
            "scope": "exact_quote_prompt_mechanism",
            "retire_if_same_verdict": True,
            "applied": len(rows) == 48 and counts["full_proposition_supported"] == 0,
            "resource_absence_is_science": False,
        },
        "activation": False,
        "production_promotion": False,
    }
    artifact["acceptance_gate_results"] = _acceptance(rows, not failures)
    artifact["field_principles"] = {
        key: "Current measured work; exposed real pilots and exact fixture oracles give only diagnostic limits."
        for key in artifact
    }
    return artifact


def independent_reduce(path: Path) -> dict[str, Any]:
    """Recompute each persisted row in a fresh process from raw responses."""
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["rows"]
    if value["verdict_class"] == "blocked":
        return {"passed": not rows and not value["model_invoked"], "calls": 0}
    panel_path = ROOT / RAW / "frozen_panel.json"
    if not panel_path.is_file():
        return {"passed": False, "reason": "frozen_panel_absent"}
    panel = {row["unit_id"]: row for row in json.loads(panel_path.read_text()).copy()}
    for row in rows:
        original = panel.get(row["unit_id"])
        if (
            original is None
            or row["source"] != original["source"]
            or row["answer"] != original["answer"]
        ):
            return {"passed": False, "reason": "source_or_answer_drift"}
        request_path, response_path = Path(row["request_path"]), Path(row["raw_response_path"])
        if (
            not request_path.is_file()
            or not response_path.is_file()
            or custody.sha256_file(request_path) != row["request_sha256"]
            or custody.sha256_file(response_path) != row["raw_response_sha256"]
        ):
            return {"passed": False, "reason": "raw_receipt_drift"}
        if json.loads(request_path.read_text()) != make_request(original, row["arm"]):
            return {"passed": False, "reason": "request_drift"}
        reply = json.loads(response_path.read_text())
        choice = reply["choices"][0]
        replay = reduce_proposal(
            original,
            row["arm"],
            str(choice["message"].get("content") or ""),
            str(choice.get("finish_reason") or "unknown"),
        )
        if replay != row["metrics"]:
            return {"passed": False, "reason": "semantic_reduction_drift"}
    return {
        "passed": True,
        "calls": len(rows),
        "independent_groups": len({row["unit_id"] for row in rows}),
    }


def cold_replay(path: Path) -> dict[str, Any]:
    """Bind the raw reducer and immutable input hashes after process restart."""
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = custody.canonical_hash(
        {
            "input_hashes": value["source_artifact_hashes"].get("producers", {}),
            "seed": SEED,
            "arms": ARMS,
            "max_tokens": 256,
            "reducer_sha256": custody.sha256_file(ROOT / CAPABILITY),
        }
    )
    reduced = independent_reduce(path)
    return {
        "passed": reduced["passed"] and expected == value["reproducibility_checksum"],
        "reduction": reduced,
    }


def _terminal_commands(root: Path, candidate: Path) -> list[validation.CommandSpec]:
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
    """Run the owned model once, then publish only a terminally checked record."""
    started = time.monotonic()
    previous.progress(started, "startup", "before", root=str(root.resolve()))
    root = root.resolve()
    if date != "20260926":
        raise SystemExit("--date must be 20260926")
    destination = output if output.is_absolute() else root / output
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7676-"))
    (root / RAW).mkdir(parents=True, exist_ok=True)
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    previous.progress(started, "preconditions", "before")
    checks, hashes, context = previous._preflight(root, started)
    failures = [check for check in checks if not check["passed"]]
    spans.append(
        previous._span("preconditions", started, begin, len(checks), len(checks) - len(failures))
    )
    previous.progress(started, "preconditions", "after", failures=len(failures))
    rows: list[dict[str, Any]] = []
    runtime: dict[str, Any] = {}
    authentication_failed = False
    if not failures:
        begin = time.monotonic()
        previous.progress(started, "freeze_panel", "before")
        try:
            pilots = [json.loads(line) for line in (root / previous.PILOT).read_text().splitlines()]
            panel = freeze_panel(pilots)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            authentication_failed = True
            failure = _gate(
                "pilot_authentication",
                "exp7602_pilot_inputs",
                str(root / previous.PILOT),
                "exact_input_hashes",
                True,
                f"{type(error).__name__}:{error}",
            )
            failures.append(failure)
            checks.append(failure)
        else:
            custody.atomic_json(root / RAW / "frozen_panel.json", panel)
            custody.atomic_json(root / RAW / "frozen_affected_validation_manifest.json", MANIFEST)
            hashes["pre_gate_receipts"]["frozen_panel"] = custody.sha256_file(
                root / RAW / "frozen_panel.json"
            )
        spans.append(
            previous._span("freeze_panel", started, begin, 24, 0 if failures else len(panel))
        )
        previous.progress(started, "freeze_panel", "after", groups=0 if failures else len(panel))
    if not failures:
        begin = time.monotonic()
        previous.progress(started, "measurement", "before", planned_calls=48)
        try:
            rows, runtime = previous._measure(
                root,
                panel,
                context,
                started,
                arms=ARMS,
                request_builder=make_request,
                response_reducer=reduce_proposal,
                raw_path=RAW,
                task_id="experiment_7676_v669_qwen_quote_relations",
            )
            runtime["model_path"] = str(context["model_path"])
            runtime["model_sha256"] = context["model_sha256"]
        except BaseException as error:
            run_dirs = sorted((root / RAW / "runs").glob(f"*-{os.getpid()}"))
            checkpoint = run_dirs[-1] / "checkpoint.json" if run_dirs else None
            if checkpoint and checkpoint.is_file():
                rows = json.loads(checkpoint.read_text())["rows"]
            runtime = {
                "model_load_attempted": 1,
                "generation_attempted": len(rows) + 1,
                "model_path": str(context["model_path"]),
                "model_sha256": context["model_sha256"],
                "device_uuid": context["selected"]["uuid"],
            }
            failure = _gate(
                "owned_generation",
                "exp7630_owned_qwen_server",
                str(root / RAW),
                "forty_eight_completed_calls",
                48,
                f"{len(rows)}:{type(error).__name__}:{error}",
            )
            failures.append(failure)
            checks.append(failure)
        spans.append(previous._span("measurement", started, begin, 48, len(rows)))
        previous.progress(
            started, "measurement", "after", completed_calls=len(rows), failure_count=len(failures)
        )
    rows_path = root / RAW / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    hashes["pre_gate_receipts"]["raw_rows"] = custody.sha256_file(rows_path)
    artifact = build_artifact(rows, failures, hashes, runtime, spans, time.monotonic() - started)
    artifact["preconditions_checked"] = checks
    if authentication_failed:
        artifact["honest_verdict"] = "complete_disqualified_input_authentication"
        artifact["verdict_class"] = "disqualified"
    if (
        rows
        and len(rows) == 48
        and any(
            row["metrics"]["full_proposition_supported"]
            for row in rows
            if row["population"] == "fixture"
        )
    ):
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = "complete_circular_positive_fixture_relation"
    previous.progress(started, "scoped_validation", "before")
    begin = time.monotonic()
    basetemp = private / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    commands = validation.build_scoped_commands(
        root,
        MANIFEST["tests"],
        MANIFEST["changed_modules"],
        static_paths=MANIFEST["static_paths"],
        basetemp=basetemp,
        coverage_file=private / ".coverage.exp7676",
    )
    receipts = validation.run_commands(
        root, commands, log_dir=private / "logs" / "scoped", heartbeat_s=45
    )
    spans.append(previous._span("scoped_validation", started, begin, len(commands), len(receipts)))
    previous.progress(
        started, "scoped_validation", "after", passed=all(row["passed"] for row in receipts)
    )
    if not all(row["passed"] for row in receipts):
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    artifact["validation_receipts"] = receipts
    candidate = private / "terminal_candidate.json"
    custody.atomic_json(candidate, artifact)
    previous.progress(started, "terminal_readers", "before", candidate=str(candidate))
    begin = time.monotonic()
    terminal = validation.run_commands(
        root,
        _terminal_commands(root, candidate),
        log_dir=private / "logs" / "terminal",
        heartbeat_s=45,
    )
    spans.append(previous._span("terminal_readers", started, begin, 4, len(terminal)))
    previous.progress(
        started, "terminal_readers", "after", passed=all(row["passed"] for row in terminal)
    )
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
    if not all(row["passed"] for row in receipts + terminal):
        artifact["honest_verdict"] = "complete_disqualified_required_checks"
        artifact["verdict_class"] = "disqualified"
        artifact["quote_pilot_measurement_complete_score"] = 0
        for result in artifact["acceptance_gate_results"].values():
            result["passed"] = False
    artifact["phase_spans"] = spans
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = {
        key: "Measured current work or explicit bounded absence." for key in artifact
    }
    custody.atomic_json(destination, artifact)
    previous.progress(
        started,
        "publish",
        "after",
        path=str(destination),
        verdict=artifact["honest_verdict"],
        sha256=custody.sha256_file(destination),
    )
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
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
