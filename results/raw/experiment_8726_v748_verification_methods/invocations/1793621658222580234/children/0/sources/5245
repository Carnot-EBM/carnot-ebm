"""REQ-REPORT-8173 / REQ-VERIFY-8173: qualify repair without inventing measurements.

The historical acquisition attempt stopped at static validation. This receipt
can qualify its runner while keeping absent scientific costs unknown.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import shared_acquisition_execution_8160 as owned
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    CommandSpec,
    build_scoped_commands,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.verify import shared_acquisition_8160 as acquisition

Json = dict[str, Any]
ROOT = acquisition.ROOT
NAME = "experiment_8173_v706_service_validation"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/reporting/service_validation_8173.py"
TEST = "tests/python/test_service_validation_8173.py"
CRASH = "tests/python/test_durable_batch_8159.py::test_crashes_deduplication"
PROTOCOL = ROOT / "openspec/change-proposals/v706-service-validation-protocol.json"
MODEL_SPECS: list[Json] = []
execute = owned.execute
atomic_json, reference, sha256_file = (
    acquisition.atomic_json,
    acquisition.reference,
    acquisition.sha256_file,
)
checksum, progress = acquisition.checksum, acquisition.progress


def validation_plan(private: Path) -> list[CommandSpec]:
    """Freeze real paths for static tools and preserve original transport assertions."""
    tests = [
        TEST,
        acquisition.TEST,
        CRASH,
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
    ]
    plan = build_scoped_commands(
        ROOT,
        tests,
        [MODULE, acquisition.OWNED[1]],
        static_paths=[CLI],
        basetemp=private,
        coverage_file=private / ".coverage",
    )
    commands = []
    for c in plan:
        args = c.argv
        if c.name in {"ruff_check", "ruff_format", "scoped_spec_coverage"}:
            args = tuple(a.split("::", 1)[0] for a in args)
        args = tuple(a + ",*/" + CLI if a.startswith("--include=") else a for a in args)
        if c.name == "changed_module_mypy":
            args += ("--strict", "--follow-imports=silent")
        commands.append(CommandSpec(c.name, args, c.scope, 600))
    return commands


def validators(path: Path) -> list[CommandSpec]:
    """Use the unchanged terminal auditors and a separate cold process."""
    py = str(ROOT / ".venv/bin/python")
    return [
        CommandSpec(
            "cold_replay", (py, "-u", str(ROOT / CLI), "--cold-replay", str(path)), "terminal", 300
        ),
        CommandSpec(
            "adversarial",
            (py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)),
            "terminal",
            300,
        ),
        CommandSpec(
            "strict_rows",
            (py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)),
            "terminal",
            300,
        ),
    ]


def inputs(root: Path, raw: Path) -> Json:
    """Bind original evidence and replay views to immutable original code bytes."""
    protocol = json.loads(PROTOCOL.read_text())
    data: Json = dict(
        checks=[], refs=[reference(PROTOCOL)], sources=[], historical_models=[], protocol=protocol
    )

    def gate(path: Path, field: str, expected: Any, observed: Any) -> None:
        data["checks"].append(
            dict(
                check=field,
                upstream=path.stem,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )

    for pin in protocol["primaries"]:
        path = root / pin["path"]
        gate(path, "resource_exists", True, path.is_file())
        if not path.is_file():
            continue
        gate(path, "sha256", pin["sha256"], sha256_file(path))
        if data["checks"][-1]["passed"] is not True:
            continue
        value = json.loads(path.read_text())
        refs = [
            reference(path),
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            value["primitive_rows"],
            value["input_data"],
        ]
        refs += [
            dict(path=r["log_path"], sha256=r["log_sha256"])
            for r in value["validation_receipts"]
            if "log_path" in r
        ]
        # The upstream capture primary binds its raw replies and original CUDA call ledger.
        capture_refs = [
            r
            for r in refs
            if Path(r["path"]).name == "experiment_8102_v701_learning_stream_capture.json"
        ]
        for ref in capture_refs:
            capture_path = Path(ref["path"])
            gate(
                capture_path,
                "capture_primary_sha256",
                ref["sha256"],
                sha256_file(capture_path) if capture_path.is_file() else None,
            )
            if not data["checks"][-1]["passed"]:
                continue
            capture = json.loads(capture_path.read_text())
            refs.extend(capture["raw_shard_hashes"])
            data["historical_models"].append(
                dict(
                    experiment_id=8102,
                    primary=ref,
                    MODEL_SPECS=capture["MODEL_SPECS"],
                    model_invocation_counts=capture["model_invocation_counts"],
                    call_ledger=capture["call_ledger"],
                    duration_s=capture["duration_s"],
                )
            )
        view = deepcopy(value)
        code = {}
        for name, digest in value["code_config_hashes"].items():
            source = Path(protocol["historical_code"].get(name, {}).get("path", str(ROOT / name)))
            gate(
                source,
                "measurement_code_sha256",
                digest,
                sha256_file(source) if source.is_file() else None,
            )
            if data["checks"][-1]["passed"]:
                target = raw / "historical_code" / (digest[7:] + source.suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                code[str(target.absolute())] = digest
                refs.append(reference(target))
        for ref in refs:
            p = Path(ref["path"])
            gate(p, "evidence_sha256", ref["sha256"], sha256_file(p) if p.is_file() else None)
        if any(not c["passed"] for c in data["checks"]):
            continue
        view["code_config_hashes"] = code
        view["reproducibility_checksum"] = checksum(view)
        replay_path = raw / f"replay_view_{pin['experiment_id']}.json"
        atomic_json(replay_path, view)
        progress("8173_historical_replay_before", len(data["sources"]), 2 - len(data["sources"]))
        passed = (acquisition.replay if pin["experiment_id"] == 8160 else acquisition.host.replay)(
            replay_path
        )
        gate(path, "historical_replay", True, passed)
        progress("8173_historical_replay_after", len(data["sources"]) + 1, 1 - len(data["sources"]))
        data["refs"].extend([*refs, reference(replay_path)])
        data["sources"].append(
            dict(
                primary=reference(path),
                value=value,
                work=json.loads(Path(value["primitive_rows"]["path"]).read_text()),
                input=json.loads(Path(value["input_data"]["path"]).read_text()),
            )
        )
    data["ready"] = all(c["passed"] for c in data["checks"])
    return data


def reduce(data: Json) -> Json:
    """Absent acquisition clocks remain unknown even when host timing is qualified."""
    source = next((s for s in data["sources"] if s["value"]["experiment_id"] == 8160), {})
    work = source.get("work", {})
    measured = bool(
        work.get("captures")
        and work.get("host_groups")
        and work.get("warmups")
        and work.get("startup_ns")
        and all("acquisition_ns" in r for r in work["captures"] if r["status"] == "completed")
    )
    if measured:
        reduction = acquisition.reduce_rows(work)
        independent = acquisition.independent_costs(reduction, work)
        return dict(
            reduction,
            composition_replay_ready_score=int(
                independent and reduction.pop("acquisition_composition_ready_score")
            ),
        )
    slots = source.get("input", {}).get("slots", [])
    rows = [
        dict(
            unit_id=f"retention-{i + 1}",
            source_cluster_id=slots[i]["source_cluster_id"]
            if i < len(slots)
            else f"missing-{i + 1}",
            arm="preserved_composition",
            condition="historical_missing_measurement",
            metric="composed_request_cost_ns",
            numerator=None,
            denominator=1,
            status="excluded",
            exclusion_reason="missing_measured_acquisition_or_service_timing",
        )
        for i in range(32)
    ]
    return dict(
        rows=rows,
        composed_cost_rows=[],
        paired_speed_intervals=[],
        zero_arithmetic_ceiling=None,
        composition_replay_ready_score=0,
        intended_count=32,
        eligible_count=0,
        independent_count=0,
        completed_count=0,
        excluded_count=32,
        censored_count=0,
        failed_count=0,
    )


def owned_checks(receipts: list[Json], fixture: bool) -> bool:
    """Receipt labels cannot substitute for required names and actual normal zero exits."""
    names = {r.get("name") for r in receipts}
    return (
        bool(receipts)
        and (fixture or set(REQUIRED_CHECK_NAMES) <= names)
        and all(
            r.get("passed") is True
            and r.get("normal_exit") is True
            and r.get("actual_exit") == 0
            and not r.get("timed_out", False)
            for r in receipts
        )
    )


def build(
    data: Json, raw: Path, receipts: list[Json], date: str, duration: float, fixture: bool = False
) -> Json:
    """Publish harness readiness separately from available preserved measurements."""
    checked = owned_checks(receipts, fixture)
    blocked = next((c["check"] for c in data["checks"] if not c["passed"]), None)
    ready = int(checked and not blocked)
    verdict = "circular_positive" if fixture else "positive"
    if blocked:
        verdict = "blocked"
    if not checked:
        verdict = "disqualified"
    reduction = reduce(data)
    atomic_json(raw / "primitive_rows.json", dict(sources=data["sources"], reduction=reduction))
    atomic_json(raw / "input_data.json", data)
    originals = [s["value"] for s in data["sources"]]
    value: Json = dict(
        experiment_id=8173,
        task_id="exp8173-service-validation",
        schema="carnot.service_validation.v1",
        honest_verdict="complete_blocked_" + str(blocked)
        if verdict == "blocked"
        else "complete_" + verdict + "_service_validation",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        claim_scope="Owned service harness qualification and historical evidence replay only; absent acquisition measurements are unknown; no deployment speed or learning claim",
        exposure_scope="exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=data["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=[],
        historical_trained_head_specs=[s.get("trained_head_specs", []) for s in originals],
        model_invocation_counts=dict(acquisition.prior.ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        historical_model_provenance=data["historical_models"]
        + [
            dict(
                experiment_id=s["experiment_id"],
                MODEL_SPECS=s["MODEL_SPECS"],
                model_invocation_counts=s["model_invocation_counts"],
                call_ledger=s["call_ledger"],
                source_artifact_hashes=s["source_artifact_hashes"],
            )
            for s in originals
        ],
        cited_upstream_artifacts=[
            dict(
                experiment_id=s["value"]["experiment_id"],
                fields_imported=[
                    "primitive_rows",
                    "input_data",
                    "validation_receipts",
                    "code_config_hashes",
                    "duration_s",
                ],
                sha256=s["primary"]["sha256"],
            )
            for s in data["sources"]
        ],
        run_date=date,
        duration_s=duration,
        random_seed=data["protocol"]["seed"],
        sample_size_budget=dict(intended_sources=32, current_model_calls=0),
        source_artifact_hashes=data["refs"],
        raw_shard_hashes=[
            reference(raw / "primitive_rows.json"),
            reference(raw / "input_data.json"),
        ],
        code_config_hashes={
            p: sha256_file(ROOT / p) for p in [MODULE, CLI, TEST, acquisition.OWNED[1]]
        },
        phase_spans=[dict(phase="repair_validation", duration_s=duration)],
        acceptance_gates=dict(
            service="Every required owned check exits normally",
            composition="All required preserved measured components and 24 independent sources",
        ),
        field_principles=dict(
            service_protocol_ready_score="Harness qualification cannot rewrite scientific history.",
            composition_replay_ready_score="Missing measured components remain unknown.",
            model_invocation_counts="Current execution is zero; imported provenance is historical.",
            independent_count="Repeated service requests do not add independent sources.",
        ),
        service_protocol_ready_score=ready,
        original_validation_failures=[
            dict(experiment_id=s["experiment_id"], **r)
            for s in originals
            for r in s["validation_receipts"]
            if not r["passed"]
        ],
        corrected_command_manifest=[
            asdict(c) for c in validation_plan(raw / "private-placeholder")
        ],
        imported_cost_rows=[
            dict(
                experiment_id=s["experiment_id"],
                rows=s.get("component_cost_rows", []),
                original_duration_s=s["duration_s"],
            )
            for s in originals
        ],
        measurement_code_hashes={
            str(s["experiment_id"]): s["code_config_hashes"] for s in originals
        },
        validation_code_hashes={
            p: sha256_file(ROOT / p) for p in [MODULE, CLI, acquisition.OWNED[1]]
        },
        original_measurement_durations=[
            dict(
                experiment_id=s["value"]["experiment_id"],
                duration_s=s["value"]["duration_s"],
                measurement_duration_s=s["work"].get("measurement_duration_s"),
                startup_ns=s["work"].get("startup_ns"),
            )
            for s in data["sources"]
        ],
        next_service_protocol=data["protocol"]["next_service"],
        fixture_mode=fixture,
        methodology="Authenticate immutable predecessor bytes; replay preserved source masks, clocks, durable service and failed static receipts with zero current model work. Missing acquisition components stay unknown.",
        **reduction,
    )
    value["composition_replay_ready_score"] *= ready
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Independently reopen custody, reductions and real validation logs."""
    progress("8173_replay_before", 0, 1)
    try:
        value = json.loads(path.read_text())
        if checksum(value) != value["reproducibility_checksum"]:
            return False
        for ref in [*value["source_artifact_hashes"], *value["raw_shard_hashes"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        if any(sha256_file(ROOT / p) != h for p, h in value["code_config_hashes"].items()):
            return False
        checked = owned_checks(value["validation_receipts"], value["fixture_mode"])
        for r in value["validation_receipts"]:
            if "log_path" in r and sha256_file(Path(r["log_path"])) != r["log_sha256"]:
                return False
        data = json.loads(Path(value["raw_shard_hashes"][1]["path"]).read_text())
        expected = reduce(data)
        expected["composition_replay_ready_score"] *= int(checked and data["ready"])
        passed = (
            all(value[k] == v for k, v in expected.items())
            and value["service_protocol_ready_score"] == int(checked and data["ready"])
            and value["required_checks_passed"] == checked
            and value["model_invocation_counts"] == dict(acquisition.prior.ZERO_INVOCATION_COUNTS)
        )
        progress("8173_replay_after", int(passed), 0)
        return passed
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(argv: list[str] | None = None) -> int:
    """Freeze argv before replay and publish only a normally validated receipt."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("8173_start", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        return 0 if replay(args.cold_replay) else 1
    output = (args.fixture_e2e or args.output).absolute()
    if args.fixture_e2e and output.is_relative_to(ROOT / "results"):
        parser.error("private fixtures require output outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-8173-validation-"))
    os.environ["COVERAGE_FILE"] = str(private / ".coverage.repository")
    plan = validation_plan(private)
    candidate = private / (NAME + ".json")
    health = CommandSpec(
        "repository_health_once",
        (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
        "separate_repository_health",
        1800,
    )
    atomic_json(
        raw / "validation_commands.json",
        dict(
            commands=[asdict(c) for c in plan],
            terminal=[asdict(c) for c in validators(candidate)],
            frozen_before_measurement=True,
            protocol=reference(PROTOCOL),
            repository_health=asdict(health),
        ),
    )
    progress("8173_preconditions_before", 0, 2)
    data = inputs(args.root, raw)
    progress("8173_preconditions_after", len(data["sources"]), 0)
    # Private direct routes execute import validation; full owned scope runs once outside fixtures.
    receipts = execute(plan[:1] if args.fixture_e2e else plan, raw)
    if not args.fixture_e2e:
        bad = (
            next(
                r
                for s in data["sources"]
                for r in s["value"]["validation_receipts"]
                if r["name"] == "ruff_check"
            )
            if data["sources"]
            else None
        )
        diagnostics = []
        if bad:
            diagnostics = execute(
                [
                    CommandSpec(
                        "original_ruff_argv",
                        tuple(bad["command_argv"]),
                        "historical_failure_reproduction",
                    )
                ],
                raw / "diagnostic",
                expected=1,
            )
        repository = execute([health], raw / "health")
    else:
        diagnostics, repository = [], []
    value = build(data, raw, receipts, args.date, time.monotonic() - began, bool(args.fixture_e2e))
    value.update(
        repository_health=repository,
        original_failure_reproduction=diagnostics,
        corrected_command_manifest=[asdict(c) for c in plan],
    )
    value["raw_shard_hashes"].append(reference(raw / "validation_commands.json"))
    value["reproducibility_checksum"] = checksum(value)
    atomic_json(candidate, value)
    if not value["required_checks_passed"]:
        atomic_json(raw / "failed_terminal_candidate.json", value)
        atomic_json(raw / "terminal_validation.json", dict(passed=False, receipts=receipts))
        progress("8173_owned_failure", len(receipts), 0)
        return 1
    progress("8173_independent_reduction_before", 0, 1)
    independent = replay(candidate)
    atomic_json(
        raw / "independent_reduction.json", dict(passed=independent, reduction=reduce(data))
    )
    progress("8173_independent_reduction_after", int(independent), 0)
    terminal = execute(validators(candidate), raw / "terminal")
    if not independent or not all(r["passed"] and r["normal_exit"] for r in terminal):
        value.update(
            service_protocol_ready_score=0,
            composition_replay_ready_score=0,
            required_checks_passed=False,
            honest_verdict="complete_disqualified_owned_validation",
            verdict_class="disqualified",
        )
        value["reproducibility_checksum"] = checksum(value)
        atomic_json(raw / "failed_terminal_candidate.json", value)
        atomic_json(raw / "terminal_validation.json", dict(passed=False, receipts=terminal))
        return 1
    value["validation_receipts"] += terminal
    value["duration_s"] = time.monotonic() - began
    value["phase_spans"] = [
        dict(phase=r["name"], duration_s=r.get("duration_s", 0)) for r in receipts + terminal
    ]
    value["raw_shard_hashes"].append(reference(raw / "independent_reduction.json"))
    value["reproducibility_checksum"] = checksum(value)

    def checked(path: Path) -> Json:
        checks = execute(validators(path), raw / "publication")
        return dict(passed=all(r["passed"] and r["normal_exit"] for r in checks), receipts=checks)

    progress("8173_publication_before", 0, 1)
    publication = publish_primary(output, value, checked)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            required_checks_passed=value["required_checks_passed"],
            normal_process_exit=True,
        ),
    )
    progress("8173_complete", len(value["validation_receipts"]), 0)
    return 0
