"""Run one bounded Qwen challenge against exact source witnesses.

The witness checks only structural claims about supplied Python source. Model
syntax and pointers remain separate observations from factual correctness.
Spec: REQ-REPORT-7651, SCENARIO-REPORT-7651-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7616_v665_evidence_schema as schema_protocol
from carnot import experiment_7617_v665_schema_pilot as transport
from carnot import experiment_7630_v666_cuda_ownership as ownership
from carnot import experiment_7631_v666_schema_pilot as prior
from carnot import experiment_7604_v664_evidence_pilot as runtime_tools
from carnot.gpu_lease_phase_journal import GpuLease, LeaseBusy, RecoveryError
from carnot.reporting import current_work_receipt as receipts
from carnot.reporting import experiment_7303_validation_scope as checks
from carnot.verify.source_claim_witness import parse_numbered_blocks, verify_claim


ROOT = Path(__file__).resolve().parents[2]
RESULT = Path("results/experiment_7651_v667_qwen_witness_challenge.json")
RAW = Path("results/raw/experiment_7651_v667_qwen_witness_challenge")
PILOT = prior.PILOT_INPUT
MODEL_ID = prior.MODEL_ID
EXPLICIT_ARM = prior.EXPLICIT_ARM
GRAMMAR_ARM = prior.CONSTRAINED_ARM
SEED = 7651
TEST = "tests/python/test_experiment_7651_v667_qwen_witness_challenge.py"
MODULE = "python/carnot/experiment_7651_v667_qwen_witness_challenge.py"
WRAPPER = "scripts/experiments/experiment_7651_v667_qwen_witness_challenge.py"
MANIFEST = {
    "requirement": "REQ-REPORT-7651",
    "tests": [TEST],
    "changed_modules": [MODULE],
    "static_paths": [WRAPPER],
}
sha256_file = receipts.sha256_file


def progress(started: float, phase: str, event: str, **values: Any) -> None:  # pragma: no cover
    """Flush a monotonic event at every operation that can hold the task."""

    print(
        json.dumps(
            {
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **values,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def gate(
    check: str, upstream: str, path: str, field: str, operator: str, expected: Any, observed: Any
) -> dict[str, Any]:
    """Keep every operand needed to reproduce a resource decision."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if operator == "eq" else False,
    }


def build_request(record: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Build a fixed pair from the canonical full text and one schema authority."""

    if arm not in (EXPLICIT_ARM, GRAMMAR_ARM):
        raise ValueError("invalid_arm")
    request = prior.build_arm_request(record, arm)
    request["seed"] = SEED
    request["max_tokens"] = 512
    request["temperature"] = 0.0
    return request


def predicate_metrics(
    source: str,
    sentences: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    *,
    closed_files: Sequence[str],
) -> dict[str, Any]:
    """Count only structural truth; prose outside the grammar remains unknown."""

    by_id = {str(row.get("response_sentence_id")): row for row in evidence}
    predicates = []
    for sentence in sentences:
        sentence_id = str(sentence["sentence_id"])
        witness = verify_claim(source, str(sentence["text"]), closed_files=list(closed_files))
        proposal = by_id.get(sentence_id, {})
        relation = proposal.get("relation", "unknown")
        truth = witness["status"] if witness["status"] != "unknown" else None
        predicates.append(
            {
                "sentence_id": sentence_id,
                "model_relation": relation,
                "independent_truth": truth,
                "source_offset": witness["source_offset"],
                "proposition_checked": witness["proposition_checked"],
                "residual_unverified_span": witness["residual_unverified_span"],
                "witness": witness,
            }
        )
    checkable = [row for row in predicates if row["independent_truth"] is not None]
    supports = [row for row in checkable if row["model_relation"] == "supports"]
    return {
        "predicates": predicates,
        "structurally_checkable_denominator": len(checkable),
        "false_support_numerator": sum(
            row["independent_truth"] == "contradicted" for row in supports
        ),
        "false_support_denominator": len(supports),
        "unknown_numerator": sum(row["independent_truth"] is None for row in predicates),
        "unknown_denominator": len(predicates),
        "residual_prose_numerator": sum(row["residual_unverified_span"] for row in predicates),
    }


def raw_rows_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Reject a changed raw response before trusting saved reduction fields."""

    return all(
        Path(str(row["raw_response_path"])).is_file()
        and sha256_file(Path(str(row["raw_response_path"]))) == row["raw_response_sha256"]
        for row in rows
    )


def blocked_artifact(failed: Sequence[Mapping[str, Any]], *, duration_s: float) -> dict[str, Any]:
    """Make external absence terminal without inventing a model invocation."""

    first = dict(failed[0])
    return {
        "honest_verdict": f"complete_blocked_{first['check']}",
        "verdict_class": "blocked",
        "model_invoked": False,
        "MODEL_SPECS": [],
        "planned_MODEL_SPECS": [MODEL_ID],
        "inference_substrate": "no_model_load",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "model_bounded_generation",
        "invocation_counts": dict(receipts.ZERO_INVOCATION_COUNTS),
        "rows": [],
        "paired_pilot_rows": [],
        "duration_s": duration_s,
        "gate_check_summary": {
            "failed_checks": [row["check"] for row in failed],
            "failed_count": len(failed),
            "first_failure": first,
        },
    }


def _complete_artifact(
    artifact: dict[str, Any],
    *,
    started: float,
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Any],
    spans: list[dict[str, Any]],
) -> None:
    """Fill common terminal fields without turning readiness into benefit."""

    artifact.update(
        schema="carnot.exp7651.v667.qwen_witness_challenge.v1",
        experiment_id="experiment_7651_v667_qwen_witness_challenge",
        milestone="2026.09.667",
        run_date="20260925",
        status="complete",
        flagged_adversarial=False,
        preconditions_checked=list(preconditions),
        planned_MODEL_SPECS=[MODEL_ID],
        planned_inference_substrate_class="model_bounded_generation",
        execution_venue="host",
        execution_venue_details={
            "owned_pid": os.getpid(),
            "device_uuid": artifact.get("gpu_lease_receipt", {}).get("device_uuid"),
        },
        phase_spans=spans,
        source_artifact_hashes=sources,
        random_seed={"paired_order_and_generation": SEED},
        sample_size_budget={
            "independent_unit": "disjoint_source_group",
            "intended": 8,
            "observed": len({row["component_hash"] for row in artifact["rows"]}),
            "eligible": len({row["component_hash"] for row in artifact["rows"]}),
            "excluded": 0,
            "censored": sum(row.get("censored", False) for row in artifact["rows"]),
            "exposure_limits": "eight groups, two requests each, one seed",
        },
        validation_receipts=artifact.get("validation_receipts", []),
        terminal_reader_outcomes=artifact.get("terminal_reader_outcomes", {}),
        verifier_is_oracle=True,
        semantic_quality_scope="Only exact structural predicates are adjudicated; other prose stays unknown.",
        qwen_challenge_complete_score=int(len(artifact["rows"]) == 16),
        duration_s=time.monotonic() - started,
    )
    valid = artifact["verdict_class"] not in ("blocked", "disqualified")
    artifact["acceptance_gate_results"] = {
        "validity": {
            "passed": valid,
            "principle": "Current raw rows and required readers govern validity.",
            "measured_operands": {"raw_rows": len(artifact["rows"])},
        },
        "readiness": {
            "passed": valid and len(artifact["rows"]) == 16,
            "principle": "A complete bounded pilot establishes feasibility only.",
            "measured_operands": {"groups": artifact["sample_size_budget"]["observed"]},
        },
        **{
            name: {
                "passed": None,
                "principle": principle,
                "measured_operands": {"independent_confirmatory_groups": 0},
            }
            for name, principle in {
                "probability_benefit": "Proper loss needs independent labels and a held-back comparator.",
                "utility": "Decision value needs typed actions and outcomes.",
                "retention": "Retained learning needs a restart and released feedback.",
                "freshness": "A fixed pilot is not fresh confirmation.",
            }.items()
        },
    }
    artifact["field_principles"] = {
        key: "Measured current work or exact upstream provenance; no downstream science gate."
        for key in artifact
    }
    artifact["reproducibility_checksum"] = receipts.canonical_hash(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def independent_reduce(path: Path) -> dict[str, Any]:
    """Replay raw tokens through the independent parser and witness checker."""

    artifact = json.loads(path.read_text(encoding="utf-8"))
    rows = artifact.get("rows", [])
    if artifact.get("verdict_class") == "blocked":
        return {"passed": not rows and not artifact["model_invoked"], "groups": 0}
    if len(rows) != 16 or not raw_rows_valid(rows):
        return {"passed": False, "reason": "raw_rows_invalid"}
    groups: dict[str, set[str]] = {}
    false_support = 0
    for row in rows:
        record = row["input_record"]
        raw = json.loads(Path(row["raw_response_path"]).read_text(encoding="utf-8"))
        content, _, finish = transport._response_parts(raw)
        parsed = schema_protocol.validate_evidence_output(record, content, finish_reason=finish)
        if parsed["accepted"] != row["parser_result"]["accepted"]:
            return {"passed": False, "reason": "pointer_validation_changed"}
        metric = predicate_metrics(
            record["complete_source"],
            record["answer_sentences"],
            parsed["evidence"],
            closed_files=row["closed_files"],
        )
        if metric != row["predicate_metrics"]:
            return {"passed": False, "reason": "predicate_reduction_changed"}
        groups.setdefault(row["component_hash"], set()).add(row["arm"])
        false_support += metric["false_support_numerator"]
    paired = len(groups) == 8 and all(
        arms == {EXPLICIT_ARM, GRAMMAR_ARM} for arms in groups.values()
    )
    return {
        "passed": paired,
        "groups": len(groups),
        "requests": len(rows),
        "checkable_false_support": false_support,
    }


def cold_replay(path: Path) -> dict[str, Any]:
    """Read one exact candidate in a new process before publication."""

    value = json.loads(path.read_text(encoding="utf-8"))
    reduced = independent_reduce(path)
    checksum = receipts.canonical_hash(
        {key: item for key, item in value.items() if key != "reproducibility_checksum"}
    )
    return {
        "passed": reduced["passed"] and checksum == value.get("reproducibility_checksum"),
        "reduction": reduced,
    }


def _preflight(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:  # pragma: no cover
    """Authenticate real inputs and changed free capacity before any load."""

    preconditions, old_hashes, context = prior.collect_preconditions(root)
    source_hashes: dict[str, Any] = {
        "producer_files": {},
        "pre_gate_receipts": {},
        "missing_inputs": [],
        "planned_outputs": [str(root / RESULT)],
    }
    for row in old_hashes:
        source_hashes["producer_files"][str(row["path"])] = row.get("sha256")
    named = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "python/carnot/experiment_7631_v666_schema_pilot.py",
        "python/carnot/experiment_7644_v667_source_witness_prototype.py",
        "python/carnot/verify/source_claim_witness.py",
        "openspec/capabilities/research-reporting/spec.md",
        "results/experiment_7631_v666_schema_pilot.json",
        "results/experiment_7644_v667_source_witness_prototype.json",
        str(PILOT),
    ]
    for relative in named:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        preconditions.append(
            gate(
                f"named_input:{Path(relative).name}",
                "declared_task_input",
                str(path),
                "readable_nonempty_file",
                "eq",
                True,
                present,
            )
        )
        if present:
            source_hashes["producer_files"][relative] = sha256_file(path)
        else:
            source_hashes["missing_inputs"].append(relative)
    witness = prior._load_json(root / "results/experiment_7644_v667_source_witness_prototype.json")
    for field, expected in (("witness_ready_score", 1), ("flagged_adversarial", False)):
        preconditions.append(
            gate(
                f"exp7644_{field}",
                "experiment_7644_v667_source_witness_prototype",
                str(root / "results/experiment_7644_v667_source_witness_prototype.json"),
                field,
                "eq",
                expected,
                witness.get(field),
            )
        )
    previous = prior._load_json(root / "results/experiment_7631_v666_schema_pilot.json")
    prior_capacity = (
        previous.get("gate_check_summary", {}).get("first_failure", {}).get("observed", {})
    )
    changed = previous.get("honest_verdict") == "complete_blocked_owned_cuda_capacity" and bool(
        prior_capacity.get("available") is False and context.get("selected_gpu") is not None
    )
    preconditions.append(
        gate(
            "changed_available_capacity",
            "experiment_7631_v666_schema_pilot",
            str(root / "results/experiment_7631_v666_schema_pilot.json"),
            "blocked_then_current_owned_capacity",
            "eq",
            True,
            changed,
        )
    )
    context["runtime_build"] = runtime_tools._runtime_build_receipt()
    preconditions.append(
        gate(
            "runtime_hash",
            "installed_llama_cpp",
            str(context["runtime_build"].get("native_library_path")),
            "native_library_sha256_present",
            "eq",
            True,
            bool(context["runtime_build"].get("native_library_sha256")),
        )
    )
    source_hashes["pre_gate_receipts"]["exp7631"] = sha256_file(
        root / "results/experiment_7631_v666_schema_pilot.json"
    )
    return preconditions, source_hashes, context


def _owned_lease(
    root: Path, device: Mapping[str, Any], model_path: str, started: float
) -> GpuLease:  # pragma: no cover
    """Wait visibly for the exact device, never joining another owner."""

    lease_root = root / RAW / "gpu_leases"
    begin = time.monotonic()
    while True:
        progress(started, "gpu_lease", "before", uuid=device["uuid"])
        try:
            lease = GpuLease.acquire(
                runtime_dir=lease_root,
                task_id="experiment_7651_v667_qwen_witness_challenge",
                device_uuid=str(device["uuid"]),
                expected_model=model_path,
                vram_before_mb=int(device["memory_used_mb"]),
                ttl_s=4800,
            )
            progress(started, "gpu_lease", "after", lease_id=lease.lease_id)
            return lease
        except (LeaseBusy, RecoveryError):
            elapsed = time.monotonic() - begin
            progress(started, "gpu_lease", "wait", elapsed_s=elapsed)
            if elapsed >= 180:
                raise LeaseBusy("lease_wait_timeout") from None
            time.sleep(min(15, 180 - elapsed))


def _measure(
    root: Path, context: Mapping[str, Any], started: float
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:  # pragma: no cover
    """Use the shipped owned server transport for exactly sixteen paired calls."""

    records = prior._load_records(root / PILOT)
    requests = [
        build_request(record, arm) for record in records for arm in (EXPLICIT_ARM, GRAMMAR_ARM)
    ]
    if any(len(json.dumps(request, ensure_ascii=False).encode()) > 20_000 for request in requests):
        raise ValueError("complete_input_overflow")
    device = context["selected_gpu"]
    lease = _owned_lease(root, device, context["model_path"], started)
    recheck = ownership.recheck_before_launch(
        str(device["uuid"]),
        [ownership._current_inventory(), ownership._current_inventory()],
        context["registry"],
    )
    if recheck["passed"] is not True:
        lease.transition("terminal_blocked")
        lease.release()
        raise RuntimeError("foreign_or_capacity_recheck_failed")
    saved = (
        transport.build_arm_request,
        transport.select_configuration,
        transport.RANDOM_SEED,
        transport.ARM_ORDER_SEED,
    )

    def request_adapter(record: Mapping[str, Any], arm: str) -> dict[str, Any]:
        return build_request(record, EXPLICIT_ARM if arm == transport.CONTROL_ARM else GRAMMAR_ARM)

    transport.build_arm_request = request_adapter
    transport.select_configuration = lambda _rows: {"selected_arm": None}
    transport.RANDOM_SEED = SEED
    transport.ARM_ORDER_SEED = SEED
    run_dir = root / RAW / "runs" / f"{int(time.time())}-{os.getpid()}"
    try:
        progress(started, "paired_generation", "before", requests=16)
        rows, runtime, invocation, _ = transport.run_owned_pilot(
            root=root,
            records=records,
            model_spec={**context["model_spec"], "gpu": int(device["index"])},
            model_sha256=context["model_sha256"],
            selected_gpu=device,
            lease=lease,
            started=started,
            run_dir=run_dir,
        )
        progress(started, "paired_generation", "after", completed=len(rows))
    finally:
        (
            transport.build_arm_request,
            transport.select_configuration,
            transport.RANDOM_SEED,
            transport.ARM_ORDER_SEED,
        ) = saved
    for row in rows:
        row["arm"] = EXPLICIT_ARM if row["arm"] == transport.CONTROL_ARM else GRAMMAR_ARM
        row["raw_response_path"] = row["response_path"]
        row["raw_response_sha256"] = row["response_sha256"]
        record = row["input_record"]
        closed = [
            block["file"]
            for block in parse_numbered_blocks(record["complete_source"])
            if block["complete"] and block["closed"]
        ]
        row["closed_files"] = closed
        parsed = schema_protocol.validate_evidence_output(
            record, row["response_text"], finish_reason=row["finish_reason"]
        )
        row["parser_result"] = parsed
        row["predicate_metrics"] = predicate_metrics(
            record["complete_source"],
            record["answer_sentences"],
            parsed["evidence"],
            closed_files=closed,
        )
        row["absolute_metric"] = row["predicate_metrics"]["false_support_numerator"]
    return rows, runtime, invocation


def _validation_commands(root: Path, private: Path) -> list[checks.CommandSpec]:  # pragma: no cover
    """Freeze changed files and give pytest a private directory with an existing parent."""

    basetemp = private / "pytest"
    basetemp.mkdir(parents=True, exist_ok=True)
    return checks.build_scoped_commands(
        root,
        MANIFEST["tests"],
        MANIFEST["changed_modules"],
        static_paths=MANIFEST["static_paths"],
        basetemp=basetemp,
        coverage_file=private / ".coverage.exp7651",
    )


def _terminal_commands(root: Path, candidate: Path) -> list[checks.CommandSpec]:  # pragma: no cover
    """Read exact unpublished bytes in independent child processes."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER)
    return [
        checks.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
        checks.CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "persisted_raw_rows",
            180,
        ),
        checks.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
        checks.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            180,
        ),
    ]


def _span(
    name: str, started: float, begin: float, planned: int, completed: int
) -> dict[str, Any]:  # pragma: no cover
    """Record disjoint elapsed work and the last completed checkpoint."""

    end = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": begin - started,
        "ended_offset_s": end - started,
        "duration_s": end - begin,
        "planned_units": planned,
        "completed_units": completed,
        "checkpoint_position": completed,
    }


def _paired_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:  # pragma: no cover
    """Keep two model arms and the conservative intersection within one group."""

    paired = []
    for group in sorted({str(row["component_hash"]) for row in rows}):
        arms = {str(row["arm"]): row for row in rows if row["component_hash"] == group}
        if set(arms) != {EXPLICIT_ARM, GRAMMAR_ARM}:
            continue
        left = arms[EXPLICIT_ARM]["predicate_metrics"]["predicates"]
        right = arms[GRAMMAR_ARM]["predicate_metrics"]["predicates"]
        intersection = sum(
            a["independent_truth"] == "supported"
            and a["model_relation"] == "supports"
            and b["model_relation"] == "supports"
            for a, b in zip(left, right)
        )
        paired.append(
            {
                "component_hash": group,
                "independent_unit": group,
                "arms": [EXPLICIT_ARM, GRAMMAR_ARM],
                "structural_control": [row["witness"] for row in left],
                "conservative_intersection_supported": intersection,
                "predicate_denominator": len(left),
                "cost_s": sum(float(arm["generation_s"]) for arm in arms.values()),
                "excluded": False,
                "censored": any(arm["censored"] for arm in arms.values()),
            }
        )
    return paired


def run_experiment(root: Path, date: str, output: Path) -> int:  # pragma: no cover
    """Authenticate, run at most one paired pilot, then publish exact checked bytes."""

    started = time.monotonic()
    progress(started, "startup", "before", root=str(root.resolve()))
    if date != "20260925":
        raise SystemExit("--date must be 20260925")
    root = root.resolve()
    destination = output if output.is_absolute() else root / output
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7651-"))
    spans: list[dict[str, Any]] = []
    begin = time.monotonic()
    progress(started, "preconditions", "before")
    preconditions, sources, context = _preflight(root)
    failures = [row for row in preconditions if row.get("passed") is not True]
    spans.append(
        _span(
            "preconditions", started, begin, len(preconditions), len(preconditions) - len(failures)
        )
    )
    progress(started, "preconditions", "after", failed=len(failures))
    if failures:
        artifact = blocked_artifact(failures, duration_s=time.monotonic() - started)
    else:
        begin = time.monotonic()
        try:
            rows, runtime, invocation = _measure(root, context, started)
        except BaseException as error:
            failure = gate(
                "owned_model_transport",
                "exp7630_owned_launcher",
                str(root / RAW),
                "sixteen_bounded_requests",
                "eq",
                True,
                f"{type(error).__name__}:{error}",
            )
            preconditions.append(failure)
            artifact = blocked_artifact([failure], duration_s=time.monotonic() - started)
            rows = []
        else:
            counts = dict(receipts.ZERO_INVOCATION_COUNTS)
            counts.update(
                model_loads_attempted=1,
                model_loads_completed=1,
                generation_calls_attempted=16,
                generation_calls_completed=16,
                forward_calls_attempted=16,
                forward_calls_completed=16,
                input_tokens=sum(int(row.get("prompt_tokens") or 0) for row in rows),
                output_tokens=sum(int(row.get("output_tokens") or 0) for row in rows),
            )
            artifact = {
                "honest_verdict": "complete_null_bounded_witness_pilot",
                "verdict_class": "null",
                "model_invoked": True,
                "MODEL_SPECS": [MODEL_ID],
                "model_specs": [runtime["model_spec"]],
                "inference_substrate": "owned_local_llama_cpp_paired_bounded_generation",
                "inference_substrate_class": "model_bounded_generation",
                "invocation_counts": counts,
                "rows": rows,
                "paired_pilot_rows": _paired_groups(rows),
                "gpu_lease_receipt": {
                    "device_uuid": runtime["gpu_uuid"],
                    "owner_pid": runtime["owner_pid"],
                    "owner_pid_start_ticks": runtime["owner_pid_start_ticks"],
                    "lease_owner": runtime["lease_owner"],
                    "free_vram_before_mb": context["selected_gpu"]["memory_free_mb"],
                    "foreign_process_checks": [
                        row for row in preconditions if row["check"] == "owned_cuda_capacity"
                    ],
                    "offload_receipt": runtime["offload_layers"],
                },
                "model_runtime_receipt": {
                    "model_sha256": context["model_sha256"],
                    "model_bytes": runtime["model_bytes"],
                    "runtime_build": runtime["runtime_build"],
                    "load_s": runtime["model_load_s"],
                    "input_tokens": counts["input_tokens"],
                    "output_tokens": counts["output_tokens"],
                    "max_output_tokens_per_request": 512,
                    "no_simulation_fallback": True,
                },
                "gate_check_summary": {"failed_checks": [], "failed_count": 0},
            }
            sources["pre_gate_receipts"]["current_invocation"] = invocation.get("receipt_checksum")
        spans.append(_span("paired_generation", started, begin, 16, len(rows)))
    progress(started, "scoped_validation", "before")
    begin = time.monotonic()
    validation = checks.run_commands(
        root,
        _validation_commands(root, private),
        log_dir=private / "logs" / "scoped",
        heartbeat_s=45,
    )
    spans.append(_span("scoped_validation", started, begin, len(validation), len(validation)))
    progress(started, "scoped_validation", "after", passed=all(row["passed"] for row in validation))
    if not all(row["passed"] for row in validation):
        artifact["honest_verdict"] = "complete_disqualified_required_validation"
        artifact["verdict_class"] = "disqualified"
    artifact["validation_receipts"] = validation
    _complete_artifact(
        artifact, started=started, preconditions=preconditions, sources=sources, spans=spans
    )
    candidate = private / "terminal_candidate.json"
    receipts.atomic_json(candidate, artifact)
    progress(started, "terminal_readers", "before", candidate=str(candidate))
    begin = time.monotonic()
    terminal = checks.run_commands(
        root,
        _terminal_commands(root, candidate),
        log_dir=private / "logs" / "terminal",
        heartbeat_s=45,
    )
    spans.append(_span("terminal_readers", started, begin, len(terminal), len(terminal)))
    progress(started, "terminal_readers", "after", passed=all(row["passed"] for row in terminal))
    artifact["validation_receipts"] += terminal
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
    if not all(row["passed"] for row in terminal):
        artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
        artifact["verdict_class"] = "disqualified"
    _complete_artifact(
        artifact, started=started, preconditions=preconditions, sources=sources, spans=spans
    )
    receipts.atomic_json(
        root / RAW / "terminal_reader_outcomes.json", artifact["terminal_reader_outcomes"]
    )
    receipts.atomic_json(destination, artifact)
    progress(started, "publish", "after", path=str(destination), sha256=sha256_file(destination))
    return 0 if artifact["verdict_class"] != "disqualified" else 1


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Dispatch production and exact read-only terminal modes."""

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
