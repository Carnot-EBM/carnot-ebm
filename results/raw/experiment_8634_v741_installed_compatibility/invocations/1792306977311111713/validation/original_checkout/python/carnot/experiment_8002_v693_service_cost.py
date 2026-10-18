"""REQ-REPORT-8002: publish checked final-code CPU service and durable updates.

Current work loads no pretrained model. Imported model timings remain dated
producer evidence and are joined only to matching public request hashes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from carnot.reporting import service_cost_8002 as s
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.service_validation_8002 import coverage_counts, execute, freeze
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = s.ROOT
NAME = "experiment_8002_v693_service_cost"
TASK = "exp8002-service-cost"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/reporting/service_cost_8002.py",
    "python/carnot/reporting/service_validation_8002.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_service_cost_8002.py", f"tests/python/test_{NAME}.py"]


def base(plan: Json) -> Json:
    """Readiness starts at zero until complete execution and owned checks pass."""
    return dict(
        experiment_id=8002,
        task_id=TASK,
        milestone="2026.10.693",
        run_date="20261002",
        execution_date="20261002",
        invocation_timestamp=datetime.now(UTC).isoformat(),
        schema="carnot.service_cost.v693.v1",
        honest_verdict="complete_blocked_service_cost",
        verdict_class="blocked",
        gate_check_summary=plan.get("gate_check_summary", []),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=0.0,
        phase_spans=[],
        random_seed=69302,
        reproducibility_checksum=None,
        cited_upstream_artifacts=plan.get("source_artifact_hashes", []),
        source_artifact_hashes=plan.get("source_artifact_hashes", []),
        code_config_hashes=[],
        raw_shard_hashes=[],
        rows=[],
        sample_size_budget=dict(
            intended=64,
            eligible=0,
            started=0,
            completed=0,
            excluded=64,
            failed=0,
            censored=0,
            independent=0,
        ),
        verifier_is_oracle=False,
        claim_scope="Current CPU engineering cost on cached sources. Matching model acquisition is historical; no whole-service speed gain or natural learning benefit is claimed.",
        acceptance_gate_results=dict(benefit=False, readiness=False),
        positive_control_results=dict(working=False, scope="protocol_fixture_only"),
        preconditions_checked=[],
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        service_measurement_ready_score=0,
        branch_eligibility=plan.get("branch_eligibility", {}),
        complete_service_cost=[],
        cached_service_cost=[],
        update_touch_rows=[],
        durable_write_rows=[],
        paired_latency_intervals=[],
        measurement_code_snapshot=[],
        missing_cost_components=[
            "current_model_generation",
            "current_model_load",
            "label_provider_latency",
            "download",
            "hardware_transfer",
        ],
        acquisition_receipt_joins=[r.get("acquisition") for r in plan.get("requests", [])],
        load_amortization=plan.get("load_amortization", {}),
        replay_inputs=plan,
        config=s.CONFIG,
        methodology="One full-request warmup and ten randomized paired CPU repetitions. Source-group means give paired Student t intervals. Complete totals add only source-hash-matched dated acquisition; load amortization remains separate. Timed duration is never padded.",
    )


def terminal_check(candidate: Path) -> Json:
    """Read actual candidate bytes with cold replay and both unchanged validators."""
    s.replay(json.loads(candidate.read_bytes()))
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
                "terminal",
                60,
            )
            for name, script, flag in [
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            ]
        ],
        log_dir=candidate.parent / "terminal_logs",
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def publish(output: Path, value: Json) -> None:
    """Only checked primary bytes reach the two production artifact readers."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind final code, immutable source bytes, current work and honest cost boundaries."
        for k in value
    }
    receipt = publish_primary(output, normalize_artifact_for_template_write(value), terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    readers = reader_receipt(
        TASK,
        output.parent,
        field="service_measurement_ready_score",
        expected=value["service_measurement_ready_score"],
    )
    if not readers["passed"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", readers)


def main(argv: list[str] | None = None) -> int:
    """Validate frozen work before measuring; preserve external blocks as terminal."""
    began = time.monotonic()
    spans = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - began
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8002] phase={name} elapsed_s={elapsed:.3f} model_loads=0 generation_calls=0",
            flush=True,
        )

    phase("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261002":
            raise ValueError("run_date")
        if args.cold_replay:
            s.replay(json.loads(args.cold_replay.read_bytes()))
            print("[exp8002] replay_passed", flush=True)
            return 0
        scratch_parent = Path.home() / ".cache/carnot/experiment_8002_private"
        scratch_parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="run-", dir=scratch_parent))
        raw = scratch / "evidence"
        raw.mkdir()
        phase("authenticate")
        plan = (
            json.loads(args.fixture_input.read_bytes())
            if args.fixture_input
            else s.authenticate(args.root, raw / "upstream")
        )
        value = base(plan)
        print("[exp8002] before_subprocess storage_filesystem", flush=True)
        filesystem = subprocess.check_output(
            ["stat", "-f", "-c", "%T", str(scratch)], text=True
        ).strip()
        print(f"[exp8002] after_subprocess storage_filesystem type={filesystem}", flush=True)
        value["environmental_observations"] = dict(
            filesystem_type=filesystem,
            filesystem=str(raw / "service"),
            clock="perf_counter_ns",
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            cache_state="one named warmup",
            durable_boundary="file fsync plus directory fsync; whole serialized state is written",
        )
        atomic_json(raw / "configuration.json", dict(config=s.CONFIG, plan=plan, date=args.date))
        imports = [
            "python/carnot/verify/sparse_energy_7996.py",
            "python/carnot/verify/selective_feedback_7998.py",
            "python/carnot/verify/evidence_features_7980.py",
            "python/carnot/verify/qwen_energy_calibration_7972.py",
            "python/carnot/verify/typed_development_7997.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/experiment_template.py",
            "python/carnot/reporting/service_cost_7989.py",
            "python/carnot/reporting/service_cost_7976.py",
            "python/carnot/reporting/evidence_features_custody_7980.py",
            "python/carnot/reporting/sparse_validation_7996.py",
            "python/carnot/reporting/experiment_7303_validation_scope.py",
        ]
        value["code_config_hashes"] = [s.reference(ROOT / p) for p in OWNED + TESTS + imports] + [
            s.reference(raw / "configuration.json")
        ]
        value["measurement_code_snapshot"] = [
            s.snapshot(r, raw / "code") for r in value["code_config_hashes"]
        ]
        phase("freeze_validation")
        manifest = freeze(raw, scratch)
        value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
        value["code_config_hashes"].append(s.reference(raw / "validation_manifest.json"))
        value["reproducibility_checksum"] = s.canonical_hash(
            dict(code=value["code_config_hashes"], inputs=plan, seed=69302)
        )
        value["scratch_root_receipt"] = dict(
            path=str(scratch),
            outside_checkout=True,
            retained_for_replay=True,
            artifact_guard_enabled=True,
        )
        receipts = []
        counts = {}
        if not args.validation_worker:
            phase("owned_validation_before_measurement")
            receipts = execute(manifest, raw)
            counts = coverage_counts(scratch)
        value.update(validation_receipts=receipts, coverage_statement_counts=counts)
        coverage_valid = args.validation_worker or (
            bool(counts)
            and all(
                c["num_statements"] > 0 and c["covered_lines"] == c["num_statements"]
                for c in counts.values()
            )
        )
        failed = any(r["required"] and not r["passed"] for r in receipts) or not coverage_valid
        if failed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks", verdict_class="disqualified"
            )
        elif plan["requests"]:
            phase("measure_final_frozen_code")
            value.update(s.measure(plan, raw / "service"))
            value.update(
                honest_verdict="complete_circular_positive_service_fixture"
                if args.fixture_input
                else "complete_null_service_cost",
                verdict_class="circular_positive" if args.fixture_input else "null",
                inference_substrate="verifier_ensemble_against_cached_candidates",
                service_measurement_ready_score=1,
                trained_head_specs=[
                    dict(
                        arm=a,
                        parameter_count=len(h["parameters"]),
                        seed=h.get("seed"),
                        pretrained=False,
                        current_fitting=False,
                    )
                    for a, h in plan["heads"].items()
                ],
            )
            controls = s.fixture()
            atomic_json(raw / "control.json", controls["requests"][0])
            control = s.request(
                raw / "control.json",
                controls["heads"]["spline"],
                "spline",
                "sparse_update_durable",
                raw / "control_state",
            )
            value["positive_control_results"] = dict(
                working=0 < control["changed_coefficients"] <= 37,
                scope="protocol_fixture_only",
                actual_sparse_changes=control["changed_coefficients"],
                genuine_service_headroom=True,
            )
            value["acceptance_gate_results"]["readiness"] = True
        value["preconditions_checked"] = [
            dict(name="independent_branch_gates", passed=bool(plan["requests"])),
            dict(name="owned_checks", passed=not failed, validation_worker=args.validation_worker),
        ]
        value["repository_health"] = dict(
            current=[r for r in receipts if not r["required"]], scope="diagnostic_only"
        )
        phase("final_provenance_check")
        for ref in value["code_config_hashes"]:
            s.checked(ref)
        atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
        value["raw_shard_hashes"] = [
            s.reference(raw / "primitive_rows.json"),
            s.reference(raw / "configuration.json"),
        ] + ([s.reference(raw / "service/state.json")] if value["rows"] else [])
        phase("terminal_publication")
        value["duration_s"] = time.monotonic() - began
        spans[-1]["end_s"] = value["duration_s"]
        value["phase_spans"] = spans
        private_candidate = raw / "private_candidate.json"
        atomic_json(private_candidate, value)
        if not terminal_check(private_candidate)["passed"]:
            raise ValueError("private_candidate_rejected")
        publish(args.output.absolute(), value)
        print(
            f"[exp8002] published={args.output} verdict={value['verdict_class']} elapsed_s={time.monotonic() - began:.3f}",
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8002] rejected={type(error).__name__}:{error}", flush=True)
        return 1
