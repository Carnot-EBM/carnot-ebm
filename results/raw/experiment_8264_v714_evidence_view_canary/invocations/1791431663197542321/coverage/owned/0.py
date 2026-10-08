"""REQ-REPORT-8264: publish actual local canary evidence after owned validation.

Current calls, private controls and historical artifacts have separate custody.
External failures block measurement; owned check failures disqualify readiness.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting import coverage_custody_8262 as custody
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.verify import evidence_view_canary_8264 as d
from carnot.verify import evidence_view_live_8264 as live
from carnot.verify import protocol_conformance_8263 as qualified
from carnot.verify.evidence_view_canary_8264 import (
    ROOT,
    NAME,
    CLI,
    plan_slots,
    private_work,
    reduce,
    forecast,
    allow_capture,
)
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TASK = live.TASK
RUN_DATE = "20261008"
MODULE = "python/carnot/verify/evidence_view_execution_8264.py"
RUNNER = MODULE
TEST = "tests/python/test_evidence_view_canary_8264.py"
OWNED = [
    MODULE,
    "python/carnot/verify/evidence_view_canary_8264.py",
    "python/carnot/verify/evidence_view_live_8264.py",
    CLI,
]
MODEL_SPECS = live.MODEL_SPECS
reference, gate = qualified.reference, qualified.gate


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Identify this invocation while exposing real work and pending counters."""
    print(f"[exp8264] phase={phase} completed={completed} pending={pending}", flush=True)


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Authenticate inputs and tools before any current neural weights load."""
    progress("before_preconditions")
    started, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence=dict(plans=[], calls=[], rosters={}, timings=[], spans={}),
        live_result={},
        tokenizer={},
        fixture=fixture,
        invocation_argv=list(sys.argv),
    )
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    with TemporaryDirectory(prefix="carnot8264-private-") as directory:
        private = Path(directory)
        probe = private / "probe"
        probe.write_bytes(b"private scratch")
        gate(
            work, probe, "private_scratch_writable", True, probe.read_bytes() == b"private scratch"
        )
        for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
            gate(
                work,
                ROOT / ".venv/bin" / name,
                "required_tool",
                True,
                (ROOT / ".venv/bin" / name).is_file(),
            )
        if fixture:
            work["evidence"] = private_work(private)
        else:
            try:
                inputs = live.public_inputs(root, work)
                if all(c["passed"] for c in work["checks"]):
                    count, work["tokenizer"] = qualified.tokenizer(work)
                    if count is not None:
                        progress("before_frozen_roster")
                        work["evidence"] = live.prepare(inputs, count)
                        gate(
                            work,
                            root / qualified.PROTOCOL,
                            "intended_eligible_canary_sources",
                            12,
                            len(work["evidence"]["plans"]),
                        )
                        progress("after_frozen_roster", len(work["evidence"]["plans"]), 0)
                        if all(c["passed"] for c in work["checks"]):
                            work["live_result"] = live.acquire(work, raw, private)
            except (OSError, ValueError, RuntimeError, TimeoutError, KeyError, TypeError) as error:
                gate(work, root, "authenticated_live_operand", "qualified", str(error))
    work.update(
        duration_s=(time.monotonic_ns() - started) / 1e9,
        clock=dict(
            started_monotonic_ns=started,
            ended_monotonic_ns=time.monotonic_ns(),
            started_wall_ns=wall,
        ),
        code_config_hashes=[
            reference(ROOT / p)
            for p in OWNED
            + [
                TEST,
                "python/carnot/verify/focal_protocol_8263.py",
                "python/carnot/verify/focal_capture_8263.py",
                "python/carnot/inference/qwen_sufficiency_7920.py",
                "python/carnot/inference/llama_cpp_process.py",
                "python/carnot/gpu_lease_phase_journal.py",
            ]
        ],
        frozen_science_sha256=qualified.SCIENCE_PIN,
        execution_contract_sha256=sha256_file(
            root / "openspec/change-proposals/v714-evidence-execution-contract.json"
        )
        if (root / "openspec/change-proposals/v714-evidence-execution-contract.json").is_file()
        else None,
    )
    atomic_json(raw / "primitive_evidence.json", work["evidence"])
    atomic_json(raw / "measurement.json", work)
    progress(
        "after_measurement", len(work["evidence"]["calls"]), 36 - len(work["evidence"]["calls"])
    )
    return work


def run_check(root: Path, spec: Json, private: Path, raw: Path, *, heartbeat_s: float = 20) -> Json:
    """Copy measured owned coverage into durable custody before scratch removal."""
    with patch.object(qualified, "OWNED", OWNED):
        return dict(qualified.run_check(root, spec, private, raw, heartbeat_s=heartbeat_s))


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze all commands and include child statements before model measurement."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[run]", "[run]\npatch = subprocess, _exit")
        + "[report]\nexclude_lines=\n"
    )
    for spec in specs["commands"]:
        if spec["name"] == "consumer_and_E2E015_019":
            spec["argv"] = [
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
            ]
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                str(ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *OWNED,
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    specs["budget"] = dict(
        measurement_s=2400, validation_s=900, implementation_closeout_s=1200, total_s=4500
    )
    return dict(specs)


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness requires owned checks and custody, never favorable effect direction."""
    failed = [c for c in work["checks"] if not c["passed"]]
    evidence = work["evidence"]
    reduced = reduce(evidence)
    binding_path = raw / "coverage_binding.json"
    binding = json.loads(binding_path.read_bytes()) if binding_path.is_file() else {}
    measured = custody.replay(binding) if binding else {}
    owned = bool(receipts) and all(r["passed"] for r in receipts) and (fixture or bool(measured))
    result = work["live_result"]
    current = dict(ZERO_INVOCATION_COUNTS)
    current.update(
        model_loads_attempted=result.get("model_loads_attempted", 0),
        model_loads_completed=result.get("model_loads_completed", 0),
        model_loads_failed=result.get("model_loads_attempted", 0)
        - result.get("model_loads_completed", 0),
        generation_calls_attempted=len(result.get("runtime_receipts", [])),
        generation_calls_completed=sum(
            r.get("response") is not None for r in result.get("runtime_receipts", [])
        ),
        generation_calls_failed=sum(
            r.get("response") is None for r in result.get("runtime_receipts", [])
        ),
    )
    telemetry = [s for r in result.get("runtime_receipts", []) for s in r["telemetry"]]
    ready = int(
        owned
        and not failed
        and not fixture
        and reduced["complete_triplets"] >= 9
        and reduced["custody_passed"]
        and bool(telemetry)
        and result.get("cleanup", {}).get("leak_free", False)
        and work["duration_s"] >= 10
    )
    projections = forecast(evidence["rosters"], evidence["timings"], evidence["spans"])
    rows = reduced.pop("rows")
    for index in range(len(evidence["plans"]), 12):
        rows.extend(
            dict(
                unit_id=f"unresolved-fit-slot-{index}",
                source_cluster_id=None,
                condition=name,
                arm="focal_grammar",
                status="censored",
                exclusion_reason="external_operands_unavailable",
                numerator=None,
                denominator=1,
                metric="syntax_yield",
                p_unsupported=None,
            )
            for name in d.VIEWS
        )
    verdict = "disqualified" if not owned else "blocked" if failed else "null"
    value = dict(
        experiment_id=8264,
        task_id=TASK,
        milestone="2026.10.714",
        run_date=RUN_DATE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (failed[0]["artifact_field"] if verdict == "blocked" else "evidence_view_canary"),
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="live_llm_inference"
        if current["model_loads_completed"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation"
        if current["model_loads_completed"]
        else "no_model_load",
        inference_mode="live_gpu_gguf" if current["model_loads_completed"] else "no_model_load",
        execution_venue="host",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=current,
        rows=rows,
        intended_count=36,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=reduced["complete_triplets"] if not fixture else 0,
        verifier_is_oracle=False,
        exposure_scope="exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=any(
            r["name"] == "adversarial_verify" and not r["passed"] for r in receipts
        ),
        acceptance_gates=dict(
            owned_validation=owned,
            changed_statement_coverage=bool(measured),
            exact_custody=reduced["custody_passed"],
            minimum_triplets=reduced["complete_triplets"] >= 9,
            live_telemetry=bool(telemetry),
            natural_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        duration_s=work["duration_s"]
        + sum(r.get("duration_s", 0) for r in receipts)
        + work.get("global_health", {}).get("duration_s", 0),
        random_seed=7138250,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[
            reference(p)
            for p in sorted(raw.rglob("*"))
            if p.name
            in {
                "measurement.json",
                "primitive_evidence.json",
                "validation_commands.json",
                "validation_receipts.json",
                "coverage_binding.json",
            }
            or p.is_file()
            and (p.parent.name in {"responses", "telemetry", "slots"} or p.name == "server.log")
        ],
        phase_spans=[dict(phase="measurement", duration_s=work["duration_s"], **work["clock"])]
        + [
            dict(
                phase=r["name"],
                duration_s=r.get("duration_s"),
                started_monotonic_ns=r.get("started_monotonic_ns"),
                ended_monotonic_ns=r.get("ended_monotonic_ns"),
            )
            for r in receipts
        ],
        cited_upstream_artifacts=work["refs"],
        view_canary_ready_score=ready,
        request_rows=evidence["calls"],
        projected_capture_seconds=projections,
        model_path_sha256=result.get("gguf_sha256"),
        server_argv=result.get("model_identity_receipt", {}).get("command", []),
        generated_tokens=sum(t.get("predicted_n", 0) for t in evidence["timings"]),
        active_gpu_telemetry=telemetry,
        service_phase_spans=dict(
            evidence["spans"],
            clocks=evidence.get("service_clocks", []),
            prefill_seconds=sum(t.get("prompt_ms", 0) for t in evidence["timings"]) / 1000,
            generation_seconds=sum(t.get("predicted_ms", 0) for t in evidence["timings"]) / 1000,
            generation_clocks=[
                {k: r[k] for k in ["started_monotonic_ns", "ended_monotonic_ns"]}
                for r in result.get("runtime_receipts", [])
            ],
        ),
        frozen_science_sha256=work["frozen_science_sha256"],
        execution_contract_sha256=work["execution_contract_sha256"],
        owned_statement_counts=measured,
        coverage_command_receipt=binding,
        tokenizer_receipt=work["tokenizer"],
        model_receipt=result,
        intended_source_count=12,
        fixture_mode=fixture,
        invocation_argv=work["invocation_argv"],
        repository_health=work.get("global_health", {}),
        response_bytes=sum(r["response_bytes"] for r in result.get("runtime_receipts", [])),
        scientific_benefit_measured=False,
        trained_head_specs=[],
        methodology_note="Frozen public focal selection and length matching; one bounded local model lifetime, exact request/view custody and independent syntax reduction. No human labels or generalization claim. Conservative maximum per-token timing forecasts include full focal rosters, startup, shutdown and retries.",
        reconciliation_note="Conductor owns ops and traceability reconciliation.",
        **reduced,
    )
    for role in ["fit", "tune", "reserved"]:
        value[role + "_capture_budget_ready_score"] = int(
            ready and projections[role]["ready_score"]
        )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind actual invocation and primitive evidence; missing operands never become measured zero."
        for k in value
    }
    for key in [
        "view_canary_ready_score",
        "fit_capture_budget_ready_score",
        "tune_capture_budget_ready_score",
        "reserved_capture_budget_ready_score",
    ]:
        value["field_principles"][key] = (
            "Execution or affordability only; no favorable effect or independent benefit is implied."
        )
    value["reproducibility_checksum"] = canonical_hash(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Cold rebuild reductions from primitives and reject even rehashed headline edits."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"] + value["active_gpu_telemetry"]:
            for prefix in ["stdout", "stderr"]:
                if (
                    prefix + "_path" in receipt
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_bytes())
        primitive = json.loads((raw / "primitive_evidence.json").read_bytes())
        if primitive != work["evidence"] or not reduce(primitive)["custody_passed"]:
            return False
        count = (lambda _: 1) if work["fixture"] else qualified.tokenizer(dict(checks=[]))[0]
        for row in primitive["plans"]:
            from copy import deepcopy

            plan = deepcopy(row["plan"])
            for view in plan["views"].values():
                view["request"].pop("rendered_prompt", None)
                view["request"].pop("rendered_input_tokens", None)
            if plan_slots([row["original"]], row["cached"], count)[0]["plan"] != plan:
                return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["fixture_mode"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Bound the real worker, validation and terminal publication in one invocation."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE], default=RUN_DATE)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.worker_output:
        measure(args.root, args.worker_output.parent)
        return 0
    fixture = args.fixture_output is not None
    output = (args.fixture_output or args.output).absolute()
    if fixture and output.is_relative_to(ROOT / "results"):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot8264-validation-") as directory:
        private = Path(directory)
        candidate = private / (NAME + ".json")
        specs = manifest(private, candidate)
        specs["measurement"] = dict(
            name="measurement",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--root",
                str(args.root),
                "--worker-output",
                str(raw / "measurement.json"),
            ],
            deadline_s=2400,
            expected_exit=0,
        )
        atomic_json(raw / "validation_commands.json", specs)
        if fixture:
            work = measure(args.root, raw, fixture=True)
            receipts = [dict(name="private_measurement_normal_exit", passed=True)]
        else:
            receipts = [run_check(ROOT, specs["measurement"], private, raw / "logs")]
            work = json.loads((raw / "measurement.json").read_bytes())
            validation_started = time.monotonic()
            for spec in specs["commands"]:
                if time.monotonic() - validation_started + spec["deadline_s"] > 900:
                    receipts.append(
                        dict(
                            name=spec["name"], passed=False, error="validation_allowance_exhausted"
                        )
                    )
                    continue
                receipts.append(run_check(ROOT, spec, private, raw / "logs"))
            work["global_health"] = run_check(
                ROOT, specs["repository_health"], private, raw / "health"
            )
            atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = build(work, raw, receipts, fixture=fixture)
        atomic_json(candidate, value)
        with (
            patch.object(execution, "e", sys.modules[__name__]),
            patch.object(execution, "run_check", run_check),
        ):
            execution.publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
