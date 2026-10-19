"""REQ-REPORT-8230: publish only terminal-checked, replayable workload boundaries.

Qualified process supervision supplies deadlines, heartbeats and complete stream
hashes. Frozen upstream data supplies arithmetic, never a current model call.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_workload_boundary_8230 as h
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.request_trace_inventory_8200 import operand
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TEST = "tests/python/test_kv260_workload_boundary_8230.py"
OWNED = [
    "python/carnot/reporting/kv260_workload_boundary_8230.py",
    "python/carnot/reporting/kv260_boundary_execution_8230.py",
]
execute = qualified.execute


def checksum(value: Json) -> str:
    """Exclude only the checksum so every other recorded field is bound."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def commands(private: Path) -> list[CommandSpec]:
    """Reuse qualified coverage of real children and private E2E consumers."""
    private.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(qualified, "MODULES", OWNED),
        patch.object(qualified, "TEST", TEST),
        patch.object(qualified.e, "CLI", h.CLI),
    ):
        plan = qualified.validation_plan(private)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    return plan


def validators(path: Path) -> list[CommandSpec]:
    """Fresh CLI replay and unchanged terminal auditors inspect exact bytes."""
    with patch.object(qualified.e, "CLI", h.CLI):
        return qualified.validators(path)


def replay(path: Path) -> Json:
    """Recompute primitives independently so even rehashed summaries cannot pass."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    primitive = json.loads(checked(value["primitive_reference"]).read_bytes())
    fresh = h.precision(data)
    if primitive["precision_rows"] != fresh:
        raise ValueError("primitive_drift")
    for key, expected in h.reduce(data, fresh).items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "kv260_boundary_ready_score",
        }:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if value["reproducibility_checksum"] != checksum(value):
        raise ValueError("checksum_drift")
    return dict(passed=True, rows_checksum=canonical_hash(fresh))


def main(argv: list[str] | None = None) -> int:
    """Freeze owned commands before arithmetic and replace only checked bytes."""
    began, wall = time.monotonic_ns(), time.time_ns()
    h.progress("start_no_model_load", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=h.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (h.NAME + ".json")).absolute()
        if args.input and output.resolve().is_relative_to((h.ROOT / "results").resolve()):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / h.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="carnot8230-"))
        private.chmod(0o700)
        candidate = private / (h.NAME + ".json")
        plan = commands(private / "checks")
        health = CommandSpec(
            "repository_health_once",
            (str(h.ROOT / ".venv/bin/pytest"), "tests/python", "-q"),
            "unrelated_repository_health",
            180,
        )
        preflight = [
            CommandSpec(
                "resources_and_scratch",
                (
                    sys.executable,
                    "-c",
                    'import pathlib,shutil,sys,pytest,coverage,ruff,mypy; p=pathlib.Path(sys.argv[1]);p.write_bytes(b"private scratch");assert p.read_bytes()==b"private scratch"; assert sys.version_info>=(3,11);print(sys.version);print(shutil.disk_usage(p.parent))',
                    str(private / "writable"),
                ),
                "preconditions",
                15,
            )
        ]
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=[asdict(s) for s in plan],
                preconditions=[asdict(s) for s in preflight],
                terminal=[asdict(s) for s in validators(candidate)],
                repository_health=asdict(health),
                config=h.CONFIG,
                frozen_before_measurement=True,
            ),
        )
        h.progress("preconditions_before", 0, 1)
        pre_receipts = execute(preflight, raw / "preflight")
        data = json.loads(args.input.read_bytes()) if args.input else h.load(args.root, raw)
        data["fixture"] = bool(args.input)
        if not all(r["passed"] for r in pre_receipts):
            data["checks"].append(
                operand("resources_and_scratch", private, "normal pass", pre_receipts)
            )
            data["cases"] = []
        pre_end = time.monotonic_ns()
        atomic_json(raw / "replay_inputs.json", data)
        h.progress("preconditions_after", 1, 0)
        measured = h.precision(data)
        measurement_end = time.monotonic_ns()
        atomic_json(raw / "primitive_rows.json", dict(precision_rows=measured))
        value = h.reduce(data, measured)
        h.progress("owned_validation_before", 0, len(plan))
        receipts = execute(plan, raw / "validation") if not args.input else []
        passed = all(r["passed"] and r["normal_exit"] for r in receipts + pre_receipts)
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                kv260_boundary_ready_score=0,
            )
        h.progress("owned_validation_after", len(receipts), 0)
        health_receipts = execute([health], raw / "health") if not args.input else []
        side = raw / "terminal_validation.json"
        code_refs: list[Json] = []
        for p in [
            *OWNED,
            h.CLI,
            TEST,
            "python/carnot/verify/utility_kernel_8221.py",
            "python/carnot/verify/utility_patch_methods_8219.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/recorder_execution_8213.py",
            "scripts/experiment_template.py",
            "ops/exclusion_manifest.yaml",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
        ]:
            frozen = h.freeze(h.ROOT / p, raw, dict(references=code_refs))
            assert frozen.is_file()
        ended = time.monotonic_ns()
        spans = [
            dict(
                phase=name,
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
                duration_s=(end - start) / 1e9,
            )
            for name, start, end in [
                ("preconditions", began, pre_end),
                ("precision", pre_end, measurement_end),
                ("validation_and_health", measurement_end, ended),
            ]
        ]
        service_path = args.root / "results/experiment_8228_v711_concurrent_service.json"
        cost_checks = [
            operand(
                "whole_request_spans.disjoint_component_ns",
                service_path,
                "authenticated complete disjoint spans",
                None if b["status"] == "unavailable" else b["observed_spans"],
            )
            for b in value["whole_request_bounds"]
            if b["status"] == "unavailable"
        ]
        value.update(
            experiment_id=8230,
            task_id="exp8230-kv260-workload-boundary",
            milestone="2026.10.711",
            run_date=args.date,
            schema="carnot.kv260_workload_boundary.v711.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=(ended - began) / 1e9,
            invocation_clocks=dict(
                started_wall_ns=wall,
                ended_wall_ns=time.time_ns(),
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
            ),
            phase_spans=spans,
            MODEL_SPECS=[],
            trained_head_specs=[
                dict(
                    kind="upstream_fixture_patch",
                    current_fit=False,
                    natural_training_qualified=False,
                )
            ]
            if measured
            else [],
            inference_substrate="verifier_ensemble_against_cached_candidates"
            if measured
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            current_model_calls=0,
            call_ledger=[],
            exposure_scope="frozen development fixtures and reused natural public membership; no independent evaluation",
            preconditions_checked=data["checks"],
            precondition_receipts=pre_receipts,
            gate_check_summary=data["checks"] + cost_checks,
            cited_upstream_artifacts=data["cited"],
            source_artifact_hashes=data["references"],
            code_config_hashes=code_refs,
            raw_shard_hashes=[reference(p) for p in sorted(raw.rglob("*")) if p.is_file()],
            replay_input_reference=reference(raw / "replay_inputs.json"),
            primitive_reference=reference(raw / "primitive_rows.json"),
            required_checks_passed=passed,
            validation_receipts=receipts,
            repository_health=health_receipts,
            fixture_mode=bool(args.input),
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(side),
            methodology="Cached frozen ordered-patch scalar replay with signed Q16.16 lookup/add and shadow FP64 fallback. Natural public probabilities are crossed with the authenticated fixture model solely for arithmetic feasibility; no natural head is fitted or deployed. Whole-request bounds require current same-request disjoint spans. Historical Qwen provenance is not a current invocation.",
            claim_scope="Host numerical boundary and historical fabric obligations only; no measured service acceleration",
            field_principles={},
        )
        value["field_principles"] = {
            key: "Bind actual invocation bytes; preserve missing operands and separate arithmetic readiness from benefit."
            for key in value
        }
        value["field_principles"].update(
            reproducibility_checksum="Canonical checksum binds every recorded field except itself.",
            precision_rows="Raw FP64/Q16.16 action differences and shadow fallback, separately for fixture and natural public rows.",
            whole_request_bounds="Hypothetical eligible lookup/add elimination only; existing fabric supports zero utility work; missing clocks remain null.",
            kv260_boundary_ready_score="One qualifies this checked operation/precision boundary, even when external service evidence is blocked.",
            branch_readiness="Retain each source's exact failed operands and original verdict without requiring sibling success.",
            trained_head_specs="Imported private fixture model only; no current fitting or qualified natural learned head.",
            repository_health="One bounded full-suite diagnostic; failures do not alter the owned check result or become a global pass.",
        )
        value = normalize_artifact_for_template_write(value)
        value["reproducibility_checksum"] = checksum(value)
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            """Publication uses real auditor exits, never readiness inferred from prose."""
            reports = execute(validators(path), raw / ("terminal-" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in reports), receipts=reports
            )

        report = validate(candidate)
        if not report["passed"]:
            atomic_json(
                raw / "failed_terminal_candidate.json",
                dict(
                    value,
                    honest_verdict="complete_disqualified_terminal_validation",
                    verdict_class="disqualified",
                    kv260_boundary_ready_score=0,
                ),
            )
            atomic_json(side, report)
            return 1
        publication = publish_primary(output, value, validate)
        atomic_json(side, dict(publication=publication))
        h.progress("published", 1, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
