"""REQ-REPORT-8244: freeze evidence and validate exact bytes before publication.

The qualified child supervisor supplies deadlines, heartbeats and stream hashes.
Cold replay reconstructs the CPU boundary without a current model or device call.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_boundary_execution_8230 as prior
from carnot.reporting import kv260_decision_boundary_8244 as h
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
TEST = "tests/python/test_kv260_decision_boundary_8244.py"
OWNED = [
    "python/carnot/reporting/kv260_decision_boundary_8244.py",
    "python/carnot/reporting/kv260_decision_execution_8244.py",
]
execute = prior.execute
checksum = prior.checksum


def commands(private: Path) -> list[CommandSpec]:
    """Reuse measured CLI-child coverage and private consumers with only owned files."""
    with (
        patch.object(prior, "OWNED", OWNED),
        patch.object(prior, "TEST", TEST),
        patch.object(prior.h, "CLI", h.CLI),
    ):
        return prior.commands(private)


def validators(path: Path) -> list[CommandSpec]:
    """Unchanged auditors inspect the same private bytes as fresh-process replay."""
    with patch.object(prior.h, "CLI", h.CLI):
        return prior.validators(path)


def replay(path: Path) -> Json:
    """Reconstruct primitives so changing a summary and its checksum still fails."""
    value = json.loads(path.read_bytes())
    if value["config"] != h.CONFIG:
        raise ValueError("configuration_drift")
    for ref in (
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"]
    ):
        checked(ref)
    data = json.loads(checked(value["replay_input_reference"]).read_bytes())
    measured = h.precision(data)
    primitive = json.loads(checked(value["primitive_reference"]).read_bytes())
    if measured != primitive["precision_rows"]:
        raise ValueError("primitive_drift")
    for key, expected in h.reduce(data, measured).items():
        if value["verdict_class"] == "disqualified" and key in {
            "honest_verdict",
            "verdict_class",
            "kv260_boundary_ready_score",
        }:
            continue
        if value[key] != expected:
            raise ValueError("reduction_drift:" + key)
    if checksum(value) != value["reproducibility_checksum"]:
        raise ValueError("checksum_drift")
    return dict(passed=True, replay_passed=True)


def main(argv: list[str] | None = None) -> int:
    """Keep commands and inputs fixed, then publish only a checked terminal result."""
    began, wall = time.monotonic_ns(), time.time_ns()
    h.progress("start_no_model_load", 0, 1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261007"], default="20261007")
    parser.add_argument("--root", type=Path, default=h.ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.output or args.root / "results" / (h.NAME + ".json")).absolute()
        fixture = bool(args.input or args.fixture_e2e)
        if fixture and output.resolve().is_relative_to((h.ROOT / "results").resolve()):
            raise ValueError("private_fixture_requires_private_output")
        raw = output.parent / "raw" / h.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="carnot8244-"))
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
                    'import pathlib,sys,shutil,pytest,coverage,ruff,mypy,numpy,scipy; p=pathlib.Path(sys.argv[1]);p.write_bytes(b"private scratch");assert p.read_bytes()==b"private scratch";print(sys.version);print(shutil.disk_usage(p.parent))',
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
        if args.input:
            h.freeze(args.input, raw, data)
        if not all(r["passed"] for r in pre_receipts):
            data["heads"] = []
            data["branches"]["heads"] = False
        pre_end = time.monotonic_ns()
        atomic_json(raw / "replay_inputs.json", data)
        h.progress("preconditions_after", 1, 0)
        measured = h.precision(data)
        measurement_end = time.monotonic_ns()
        atomic_json(raw / "primitive_rows.json", dict(precision_rows=measured))
        value = h.reduce(data, measured)
        h.progress("owned_validation_before", 0, len(plan))
        receipts = execute(plan, raw / "validation") if not fixture else []
        passed = all(r["passed"] and r["normal_exit"] for r in receipts + pre_receipts)
        if not passed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                kv260_boundary_ready_score=0,
            )
        h.progress("owned_validation_after", len(receipts), 0)
        health_receipts = execute([health], raw / "health") if not fixture else []
        coverage_files = [
            p
            for p in (private / "checks").glob("*")
            if p.is_file() and (p.name.startswith(".coverage") or p.name == "coverage.ini")
        ]
        for path in coverage_files:
            destination = raw / "coverage_evidence" / path.name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, destination)
        side = raw / "terminal_validation.json"
        code_refs: list[Json] = []
        for path in [
            *OWNED,
            h.CLI,
            TEST,
            "python/carnot/verify/margin_energy_training_8237.py",
            "python/carnot/verify/restricted_action_rule_8207.py",
            "python/carnot/verify/evidence_energy_8154.py",
            "python/carnot/reporting/hardware_workload_obligations_8216.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/recorder_execution_8213.py",
            "scripts/experiment_template.py",
            "ops/exclusion_manifest.yaml",
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "ops/e2e-test-plan.md",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "research-hardware-wishlist.md",
            "ops/hardware-bringup-prep.md",
            "docs/research-notes/v712-kv260-decision-boundary.md",
        ]:
            h.freeze(h.ROOT / path, raw, dict(references=code_refs))
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
        value.update(
            experiment_id=8244,
            task_id="exp8244-kv260-decision-boundary",
            milestone="2026.10.712",
            run_date=args.date,
            schema="carnot.kv260_decision_boundary.v712.v1",
            config=h.CONFIG,
            random_seed=h.CONFIG["seed"],
            duration_s=(ended - began) / 1e9,
            phase_spans=spans,
            invocation_clocks=dict(
                started_wall_ns=wall,
                ended_wall_ns=time.time_ns(),
                started_monotonic_ns=began,
                ended_monotonic_ns=ended,
            ),
            invocation_argv=list(argv if argv is not None else sys.argv[1:]),
            MODEL_SPECS=[],
            inference_substrate="verifier_ensemble_against_cached_candidates"
            if measured
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            current_model_calls=0,
            call_ledger=[],
            exposure_scope="reused public development sources; dependent head/precision arms; no independent generalization",
            preconditions_checked=data["checks"],
            precondition_receipts=pre_receipts,
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
            methodology="CPU coefficient-storage shadow evaluation through the existing margin scorer, basis, restricted action rule and sigmoid slope envelope. Signed symmetric 8/16-bit weights; temperature and bases remain float64. Raw probability/action/false-accept changes are retained, with exact CPU fallback near thresholds. Request bounds remove the complete scoring envelope but retain issue-to-durability, cold and failed-request work. Historical Qwen and SSH provenance are not current calls.",
            claim_scope="software numerical sensitivity and optimistic whole-request bounds; no measured hardware speedup",
            field_principles={},
        )
        value["field_principles"] = {
            key: "Bind actual invocation bytes and preserve unavailable operands; execution readiness is separate from scientific benefit."
            for key in value
        }
        value["field_principles"].update(
            precision_rows="One original source/head/precision row; coefficient quantization only with unchanged float64 CPU nonlinear operations.",
            precision_source_summaries="Dependent rows reduced within original public source; labels only count false accepts, never enter scoring.",
            whole_request_bound="Optimistic scoring-free bounds retain queue, acquisition and durable work; cold-inclusive summed work is not concurrency throughput.",
            measured_service_spans="Authenticated Exp8242 request outcomes and clocks; failed requests remain excluded from completion gains.",
            kv260_boundary_ready_score="Checked boundary execution and historical custody; no head benefit or device implementation implied.",
            kv260_obligation="Authenticated historical quadratic Ising k_max<=5 and future SSH obligation, independent of sibling scientific outcomes.",
            trained_head_specs="Imported Exp8237 learned heads only; current fitting and generator updates are zero.",
            repository_health="Single bounded full-suite diagnostic; its failures never become a global pass.",
            reproducibility_checksum="Canonical checksum binds every field except itself.",
        )
        value = normalize_artifact_for_template_write(value)
        value["reproducibility_checksum"] = checksum(value)
        atomic_json(candidate, value)

        def validate(path: Path) -> Json:
            """Real validator exits determine publication, never readiness prose alone."""
            reports = execute(validators(path), raw / ("terminal-" + str(time.time_ns())))
            return dict(
                passed=all(r["passed"] and r["normal_exit"] for r in reports), receipts=reports
            )

        report = validate(candidate)
        if not report["passed"]:
            failed = dict(
                value,
                honest_verdict="complete_disqualified_terminal_validation",
                verdict_class="disqualified",
                kv260_boundary_ready_score=0,
            )
            failed["reproducibility_checksum"] = checksum(failed)
            atomic_json(raw / "failed_terminal_candidate.json", failed)
            atomic_json(side, report)
            return 1
        if output.exists():
            h.freeze(output, raw / "historical_primary", dict(references=[]))
        publication = publish_primary(output, value, validate)
        atomic_json(side, dict(publication=publication))
        h.progress("published", 1, 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        return 1
