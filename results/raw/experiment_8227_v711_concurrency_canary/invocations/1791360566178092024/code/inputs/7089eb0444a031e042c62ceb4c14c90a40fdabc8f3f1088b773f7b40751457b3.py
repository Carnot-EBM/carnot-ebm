"""REQ-REPORT-8227: publish current canary readiness independently of benefit.

Private checks and saved primitive bytes precede the unchanged atomic publisher.
An external block is a complete result, while an owned validation failure cannot
qualify readiness. Historical primary files are only copied and authenticated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.inference import concurrency_runtime_8227 as runtime
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import publish_primary, validate_primary
from carnot.reporting.request_trace_inventory_8200 import copy_bytes, operand
from carnot.verify import concurrency_canary_8227 as e
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
execute, CommandSpec, checksum = qualified.execute, qualified.CommandSpec, qualified.checksum
PROTOCOL = "openspec/change-proposals/v711-concurrent-acquisition-protocol.json"
UPSTREAM = "results/experiment_8214_v709_prospective_service_measurement.json"
NAMED = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "python/carnot/verify/request_recorder_8213.py",
    "python/carnot/verify/prospective_service_8214.py",
    "python/carnot/reporting/prospective_service_execution_8214.py",
    "tests/python/test_prospective_request_recorder_8213.py",
    "python/carnot/inference/sota_models.py",
]


def validation_plan(private: Path) -> list[CommandSpec]:
    """Reuse measured child coverage and scoped checks with only this task's files."""
    with (
        patch.object(qualified, "MODULES", e.MODULES),
        patch.object(qualified, "TEST", e.TEST),
        patch.object(qualified.e, "CLI", e.CLI),
    ):
        plan = qualified.validation_plan(private)
    plan.append(
        CommandSpec(
            "e2e022",
            (
                str(e.ROOT / ".venv/bin/pytest"),
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "e2e022"),
                "tests/python/test_prospective_request_recorder_8213.py",
                "-q",
            ),
            "private_e2e",
            180,
        )
    )
    return plan


def validators(path: Path) -> list[CommandSpec]:
    """The unchanged auditors operate on a private candidate and actual replay CLI."""
    with patch.object(qualified.e, "CLI", e.CLI):
        return qualified.validators(path)


def build(
    data: Json, result: Json, raw: Path, receipts: list[Json], duration: float, fixture: bool
) -> Json:
    """Retain missing calls and count distinct paired sources without accuracy credit."""
    protocol = data.get("protocol", {})
    schedule = protocol.get("canary", []) or [
        dict(
            request_id=f"unbound-{i}",
            source_cluster_id=None,
            arm="serial" if i < 4 else "concurrent",
        )
        for i in range(8)
    ]
    actual = {r["request_id"]: r for r in result.get("rows", [])}
    rows = []
    for slot in schedule:
        primitive = actual.get(slot["request_id"], {})
        status = primitive.get("status", "censored")
        complete = status == "completed"
        observed_response = primitive.get("result", {})
        if status == "error":
            observed_response = observed_response.get("response", {})
        rows.append(
            dict(
                request_id=slot["request_id"],
                source_cluster_id=slot["source_cluster_id"],
                arm=slot["arm"],
                condition="original",
                seed=e.SEED,
                status=status,
                completed=complete,
                failed=status == "error",
                censored=status == "censored",
                excluded=False,
                missing_status=status == "censored",
                metric="durable_isolated_completion",
                numerator=int(complete) if status != "censored" else None,
                denominator=1,
                clocks=primitive.get("clocks"),
                slot=primitive.get("slot"),
                completion_evidence_score=int(complete),
                output_token_count=observed_response.get("usage", {}).get("completion_tokens"),
            )
        )
    sources = sorted({r["source_cluster_id"] for r in rows if r["source_cluster_id"] is not None})
    pairs = [
        dict(
            source_cluster_id=s,
            request_ids=[r["request_id"] for r in rows if r["source_cluster_id"] == s],
            completed=sum(r["completed"] for r in rows if r["source_cluster_id"] == s) == 2,
        )
        for s in sources
    ]
    completed_pairs = sum(p["completed"] for p in pairs)
    checks = [*data["checks"], *result.get("checks", [])]
    failures = [c for c in checks if not c["passed"]]
    owned = all(r["passed"] and r.get("normal_exit", True) for r in receipts)
    isolation = result.get("qualification", {}).get("passed", False)
    generation_s = sum(
        (s["end_ns"] - s["start_ns"]) / 1e9
        for s in result.get("phase_spans", [])
        if s["phase"] == "generation"
    )
    slots_ok = {r["slot"] for r in rows if r["completed"]} == {0, 1}
    live = not fixture and len([r for r in result.get("loads", []) if r["completed"]]) == 2
    ready = int(
        owned
        and not failures
        and isolation
        and live
        and slots_ok
        and completed_pairs >= 3
        and generation_s >= 10
    )
    cls, verdict = "null", "complete_null_canary_not_qualified"
    if ready or fixture:
        cls, verdict = "circular_positive", "complete_circular_positive_request_isolation"
    if failures:
        suffix = re.sub("[^a-z0-9_]", "_", failures[0]["artifact_field"].lower())
        cls, verdict = "blocked", "complete_blocked_" + suffix
    if not owned or (result.get("qualification") and not isolation):
        cls, verdict, ready = "disqualified", "complete_disqualified_owned_validation", 0
    counts = dict(ZERO_INVOCATION_COUNTS)
    if not fixture:
        counts.update(
            model_loads_attempted=len(result.get("loads", [])),
            model_loads_completed=sum(r["completed"] for r in result.get("loads", [])),
            model_loads_failed=sum(not r["completed"] for r in result.get("loads", [])),
            generation_calls_attempted=sum(
                r["status"] != "censored" for r in result.get("rows", [])
            ),
            generation_calls_completed=sum(r["completed"] for r in rows),
            generation_calls_failed=sum(r["failed"] for r in rows),
        )
    counts["generation_calls"] = counts["generation_calls_attempted"]
    value: Json = dict(
        experiment_id=8227,
        experiment=8227,
        task_id=e.TASK,
        milestone="2026.10.711",
        run_date="20261007",
        schema="carnot.v711.concurrency-canary.v1",
        title="Bounded two-slot Qwen canary",
        honest_verdict=verdict,
        verdict_class=cls,
        gate_check_summary=failures,
        inference_substrate="live_llm_inference"
        if counts["model_loads_attempted"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="model_bounded_generation"
        if counts["generation_calls_attempted"]
        else "model_load_no_generation"
        if counts["model_loads_attempted"]
        else "blocked_no_run"
        if failures
        else "no_model_load",
        declared_live_substrate="live_llm_inference",
        declared_live_substrate_class="model_bounded_generation",
        MODEL_SPECS=e.MODEL_SPECS,
        model_invocation_counts=counts,
        rows=rows,
        intended_count=8,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=completed_pairs,
        verifier_is_oracle=True,
        exposure_scope="public_fit_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        flagged_adversarial=False,
        required_checks_passed=owned,
        concurrent_canary_ready_score=ready,
        acceptance_gates=dict(
            required_inputs=not failures,
            owned_validation=owned,
            request_isolation=isolation,
            real_qwen_slots=live and slots_ok,
            three_complete_pairs=completed_pairs >= 3,
            generation_floor=generation_s >= 10,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_receipt.json"),
        preconditions_checked=checks,
        duration_s=duration,
        random_seed=e.SEED,
        phase_spans=result.get("phase_spans", []),
        source_artifact_hashes=data.get("refs", []),
        code_config_hashes=data.get("code", []),
        raw_shard_hashes=[],
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=[
                    "source identities",
                    "model/template identity",
                    "historical disposition",
                ],
            )
            for r in data.get("refs", [])
        ],
        concurrent_protocol_path=data.get("protocol_path"),
        concurrent_protocol_sha256=data.get("protocol_sha256"),
        canary_pairs=pairs,
        model_path_sha256=protocol.get("identity", {}).get("model_sha256"),
        gpu_lease=result.get("gpu_lease", {}),
        server_argv=protocol.get("server_argv", {}),
        context_per_slot=4096,
        generated_tokens=sum(r["output_token_count"] or 0 for r in rows),
        measured_generation_s=generation_s,
        actual_launch_order=result.get("actual_launch_order", []),
        future_benchmark_intended_count=len(protocol.get("benchmark", [])),
        deployment_demand_observed=False,
        measured_repeat_frequency=None,
        nfr01_met=False,
        benefit_claim=False,
        methodology_note="Independent requests change client concurrency only. Four exposed fit sources qualify transport readiness; "
        "they establish no accuracy, latency improvement, deployment benefit or independent generalization. Fixture evidence is circular.",
        replay_inputs=dict(
            data_path=str(raw / "data.json"), result_path=str(raw / "result.json"), fixture=fixture
        ),
    )
    value["field_principles"] = {
        k: "Bind current invocation bytes; preserve missing operands and separate readiness from benefit."
        for k in [*value, "reproducibility_checksum", "repository_health"]
    }
    for names, principle in [
        (
            "experiment_id experiment task_id milestone run_date schema title",
            "Identify this producer, protocol version and actual invocation date.",
        ),
        (
            "honest_verdict verdict_class gate_check_summary",
            "Terminal outcome follows exact authenticated blockers and owned failures.",
        ),
        (
            "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts declared_live_substrate declared_live_substrate_class",
            "Actual calls select the duration class; cached provenance earns zero calls.",
        ),
        (
            "rows intended_count completed_count failed_count censored_count excluded_count independent_count canary_pairs",
            "Keep every arm obligation and distinguish repeated calls from independent source pairs.",
        ),
        (
            "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score benefit_claim methodology_note",
            "Development and oracle-defined readiness cannot establish generalization or scientific benefit.",
        ),
        (
            "flagged_adversarial required_checks_passed acceptance_gates validation_receipts terminal_validation_sidecar_path",
            "Current owned checks and byte-bound terminal receipts govern safe publication.",
        ),
        (
            "preconditions_checked duration_s random_seed reproducibility_checksum source_artifact_hashes code_config_hashes raw_shard_hashes phase_spans replay_inputs cited_upstream_artifacts",
            "Preserve exact inputs, code, primitive bytes, commands and observed work clocks for cold replay.",
        ),
        (
            "concurrent_canary_ready_score",
            "One requires qualified request isolation, genuine Qwen calls and both observed slots.",
        ),
        (
            "concurrent_protocol_path concurrent_protocol_sha256 model_path_sha256 gpu_lease server_argv context_per_slot actual_launch_order future_benchmark_intended_count",
            "Bind sealed workload, two-slot context, model bytes, ownership and future launch obligations before performance evidence.",
        ),
        (
            "generated_tokens measured_generation_s",
            "Retain tokens and current measured generation spans, including failed completions.",
        ),
        (
            "deployment_demand_observed measured_repeat_frequency nfr01_met repository_health",
            "Designed canary work and bounded repository diagnostics supply no deployment or global-pass claim.",
        ),
    ]:
        value["field_principles"].update({name: principle for name in names.split()})
    return value


def replay(path: Path) -> bool:
    """Reopen every primitive hash before reconstructing headline counts in a child."""
    try:
        value = json.loads(path.read_text())
        validate_primary(value, path)
        if value["reproducibility_checksum"] != checksum(value):
            return False
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            bound = Path(ref.get("frozen_path", ref["path"]))
            if e.sha256_file(bound) != ref["sha256"]:
                return False
        inputs = value["replay_inputs"]
        data = json.loads(Path(inputs["data_path"]).read_text())
        result = json.loads(Path(inputs["result_path"]).read_text())
        for row in result.get("rows", []):
            journal = e.recorder.Journal(Path(row["journal_path"]))
            terminal = next(
                r
                for r in journal.events
                if r["request_id"] == row["request_id"] and r["event"] == "terminal"
            )
            if terminal["status"] != row["status"] or terminal["result"] != row["result"]:
                return False
            clocks = row["clocks"]
            if not (
                clocks["issue"]
                == terminal["issued_monotonic_ns"]
                <= clocks["queue"]
                <= clocks["end"]
                <= terminal["observed_monotonic_ns"]
                <= clocks["durability"]
            ):
                return False
        rebuilt = build(
            data,
            result,
            Path(inputs["data_path"]).parent,
            value["validation_receipts"],
            value["duration_s"],
            inputs["fixture"],
        )
        return all(value[k] == rebuilt[k] for k in rebuilt if k != "raw_shard_hashes")
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def main(argv: list[str] | None = None) -> int:
    """Seal commands before work and expose only normally validated terminal bytes."""
    began = time.monotonic()
    e.progress("8227_start", 0, 8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        e.progress("8227_cold_replay_after", int(passed), 0)
        return 0 if passed else 1
    fixture = args.fixture_e2e is not None
    output = (args.fixture_e2e or args.output).absolute()
    if fixture and output.resolve().is_relative_to(e.ROOT / "results"):
        parser.error("private fixture output must remain outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8227-validation-"))
    private.chmod(0o700)
    candidate = private / (e.NAME + ".json")
    plan = validation_plan(private)
    health = CommandSpec(
        "repository_health_once",
        (str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_health_not_owned",
        600,
    )
    e.atomic_json(
        raw / "validation_commands.json",
        dict(
            commands=[asdict(c) for c in plan],
            terminal=[asdict(c) for c in validators(candidate)],
            repository_health=asdict(health),
            frozen_before_measurement=True,
        ),
    )
    e.progress("8227_inputs_before", 0, len(NAMED))
    data = qualified.inputs(args.root, raw / "authenticated")
    for name in [*NAMED, UPSTREAM]:
        path = args.root / name
        data["checks"].append(
            operand("named_input_" + path.stem, path, True, True if path.exists() else None)
        )
        if path.is_file():
            data["refs"].append(copy_bytes(path, raw / "named_inputs"))
            if name == UPSTREAM:
                try:
                    v = json.loads(path.read_text())
                    good = (
                        v["experiment_id"] == 8214
                        and v["verdict_class"] == "null"
                        and v["required_checks_passed"] is True
                    )
                except (ValueError, KeyError, TypeError):
                    good = False
                data["checks"].append(operand("exp8214_schema", path, True, good))
    for check in data["checks"]:
        if not Path(check["path"]).exists():
            check["observed"] = None
    data["code"] = [copy_bytes(e.ROOT / p, raw / "code") for p in [*e.MODULES, e.CLI, e.TEST]]
    data["ready"] = all(c["passed"] for c in data["checks"])
    e.progress("8227_inputs_after", int(data["ready"]), 0)
    result: Json = {}
    if data["ready"]:
        data["protocol"] = e.freeze(data["roster"], data["identity"])
        if not fixture:
            e.progress("8227_resources_before", 0, 1)
            resources = runtime.preflight(data["identity"], raw)
            data["checks"].extend(resources["checks"])
            data["resource_receipts"] = resources["receipts"]
            e.atomic_json(raw / "resources.json", resources)
            e.progress(
                "8227_resources_after", int(all(c["passed"] for c in resources["checks"])), 0
            )
            if all(c["passed"] for c in resources["checks"]):
                protocol = data["protocol"]
                model, gpu = Path(resources["model"]["model_path"]), resources["gpu"]["index"]
                names = ["canary_serial", "canary_concurrent", *protocol["launch_order"]]
                protocol["server_argv"] = {
                    name: runtime.command(model, raw / name, gpu) for name in names
                }
        protocol_path = (
            raw / "protocol.json" if fixture or args.root != e.ROOT else e.ROOT / PROTOCOL
        )
        e.atomic_json(protocol_path, data["protocol"])
        data["protocol_path"], data["protocol_sha256"] = (
            str(protocol_path),
            e.sha256_file(protocol_path),
        )
        data["refs"].append(copy_bytes(protocol_path, raw / "sealed_protocol"))
    receipts = execute(plan[:1] if fixture or args.root != e.ROOT else plan, raw / "validation")
    if (
        data["ready"]
        and all(c["passed"] for c in data["checks"])
        and all(r["passed"] for r in receipts)
    ):
        e.progress("8227_isolation_before", 0, 1)
        qualification = e.qualify(raw / "isolation")
        result["qualification"] = qualification
        e.progress("8227_isolation_after", int(qualification["passed"]), 0)
        if qualification["passed"] and not fixture:
            os.environ["CARNOT_FORCE_LIVE"] = "1"
            result.update(runtime.live(data["protocol"], resources, raw / "canary"))
    for row in result.get("rows", []):
        row["journal_path"] = str(
            raw / "canary" / ("canary_" + row["arm"]) / "requests/events.jsonl"
        )
    e.atomic_json(raw / "data.json", data)
    e.atomic_json(raw / "result.json", result)
    value = build(data, result, raw, receipts, time.monotonic() - began, fixture)
    value["repository_health"] = (
        execute([health], raw / "health") if not fixture and args.root == e.ROOT else []
    )
    value["raw_shard_hashes"] = [
        e.recorder.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()
    ]
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = checksum(value)
    e.atomic_json(candidate, value)
    terminal = execute(validators(candidate), raw / "terminal")
    if not all(r["passed"] for r in terminal):
        e.atomic_json(raw / "failed_terminal_candidate.json", value)
        return 1
    e.atomic_json(
        raw / "terminal_receipt.json",
        dict(receipts=terminal, candidate_sha256=e.sha256_file(candidate)),
    )
    checked_hash = e.sha256_file(candidate)
    publication = publish_primary(
        output, value, lambda p: dict(passed=e.sha256_file(p) == checked_hash, receipts=terminal)
    )
    e.atomic_json(raw / "publication_receipt.json", publication)
    e.progress("8227_published", 8, 0)
    return 0
