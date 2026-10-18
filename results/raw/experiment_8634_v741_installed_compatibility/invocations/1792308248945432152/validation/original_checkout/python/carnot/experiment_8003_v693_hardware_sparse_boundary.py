"""REQ-REPORT-8003: qualify CPU arithmetic and retain authenticated board obligations.

The deliverable declares no current model or board invocation. Owned checks
and final-byte validators must finish before the primary becomes reader-visible.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import hardware_sparse_8003 as h
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.sparse_validation_7996 import execute
from carnot.reporting.validation_8003 import coverage_counts, freeze, normalize_receipts
from carnot.verify.fixedpoint_sparse_8003 import CONFIG
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = h.ROOT
NAME = "experiment_8003_v693_hardware_sparse_boundary"
TASK = "exp8003-hardware-sparse-boundary"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/reporting/hardware_sparse_8003.py",
    "python/carnot/verify/fixedpoint_sparse_8003.py",
    "python/carnot/reporting/validation_8003.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_hardware_sparse_8003.py", f"tests/python/test_{NAME}.py"]


def terminal_check(candidate: Path) -> Json:
    """Both unchanged validators read the actual private candidate bytes."""
    reduced = h.replay(json.loads(candidate.read_bytes()))
    receipts = run_commands(
        ROOT,
        [
            CommandSpec(
                name,
                (str(ROOT / ".venv/bin/python"), "-u", script, flag, str(candidate)),
                "terminal",
                60,
            )
            for name, script, flag in (
                ("adversarial", "scripts/adversarial_verify.py", "--json"),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
            )
        ],
        log_dir=candidate.parent / "terminal_logs",
        heartbeat_s=30,
    )
    flagged = json.loads(Path(receipts[0]["log_path"]).read_text())["flagged_count"]
    return dict(
        passed=not flagged and all(r["passed"] for r in receipts),
        flagged_adversarial=bool(flagged),
        receipts=receipts,
        candidate_sha256=sha256_file(candidate),
        cold_reduction=reduced,
    )


def publish(output: Path, value: Json, scratch: Path) -> None:
    """Check private bytes first, then expose exactly those bytes to both readers."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        key: "Separate current CPU emulation, dated immutable producer evidence, terminal gates and estimates."
        for key in value
    }
    candidate = scratch / "candidate" / output.name
    atomic_json(candidate, normalize_artifact_for_template_write(value))
    report = terminal_check(candidate)
    atomic_json(scratch / "terminal_validation.json", report)
    if not report["passed"]:
        raise ValueError("candidate_rejected_private")
    private_readers = reader_receipt(
        TASK,
        candidate.parent,
        field="hardware_evidence_ready_score",
        expected=value["hardware_evidence_ready_score"],
    )
    if (
        not private_readers["passed"]
        or private_readers["gate_sha256"] != report["candidate_sha256"]
    ):
        raise ValueError("private_primary_resolution")

    def same_bytes(path: Path) -> Json:
        """Publication cannot substitute bytes after the terminal checks finish."""
        if sha256_file(path) != report["candidate_sha256"]:
            raise ValueError("publication_drift")
        return report

    receipt = publish_primary(output, json.loads(candidate.read_bytes()), same_bytes)
    atomic_json(raw / "terminal_validation.json", receipt)
    readers = reader_receipt(
        TASK,
        output.parent,
        field="hardware_evidence_ready_score",
        expected=value["hardware_evidence_ready_score"],
    )
    if not readers["passed"] or readers["gate_sha256"] != receipt["primary_sha256"]:
        raise ValueError("primary_resolution")
    atomic_json(raw / "primary_resolution.json", readers)


def main(argv: list[str] | None = None) -> int:
    """Freeze commands before measurement and retain complete terminal outcomes."""
    began = time.monotonic()
    spans: list[Json] = []

    def phase(name: str) -> None:
        elapsed = time.monotonic() - began
        if spans:
            spans[-1]["end_s"] = elapsed
        spans.append(dict(phase=name, start_s=elapsed))
        print(
            f"[exp8003] phase={name} elapsed_s={elapsed:.3f} pretrained_loads=0 generation_calls=0",
            flush=True,
        )

    phase("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--repository-health-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.date != "20261002":
            raise ValueError("run_date")
        if args.cold_replay:
            h.replay(json.loads(args.cold_replay.read_bytes()))
            print("[exp8003] replay_passed", flush=True)
            return 0
        parent = Path.home() / ".cache/carnot/experiment_8003_private"
        parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
        raw = scratch / "evidence"
        raw.mkdir()
        phase("freeze_configuration_and_commands")
        atomic_json(
            raw / "configuration.json",
            dict(
                config=CONFIG,
                producer_pins=h.PINS,
                date=args.date,
                model_specs=[],
                inference_substrate_class="no_model_load",
            ),
        )
        manifest = freeze(raw, scratch, args.repository_health_receipt)
        imports = [
            "python/carnot/verify/sparse_energy_7996.py",
            "python/carnot/verify/typed_development_7997.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/sparse_validation_7996.py",
            "python/carnot/reporting/experiment_7303_validation_scope.py",
            "scripts/experiment_template.py",
        ]
        refs = [
            dict(path=str(ROOT / name), sha256=sha256_file(ROOT / name))
            for name in OWNED + TESTS + imports
        ] + [
            dict(path=str(path), sha256=sha256_file(path))
            for path in (raw / "configuration.json", raw / "validation_manifest.json")
        ]
        phase("authenticate_immutable_producers")
        plan = (
            json.loads(args.fixture_input.read_bytes())
            if args.fixture_input
            else h.authenticate(args.root, raw / "upstream")
        )
        atomic_json(raw / "replay_inputs.json", plan)
        receipts = [] if args.validation_worker else execute(manifest, raw)
        normalize_receipts(receipts, manifest)
        if args.repository_health_receipt:
            previous = json.loads(args.repository_health_receipt.read_bytes())
            health = next(r for r in previous["rows"] if r["name"] == "full_pytest")
            receipts.append(
                dict(
                    health,
                    required=False,
                    reused_diagnostic=True,
                    original_receipt_path=str(args.repository_health_receipt),
                )
            )
        counts = {} if args.validation_worker else coverage_counts(scratch)
        phase("before_cpu_emulation")
        value = h.reduce(plan)
        phase("after_cpu_emulation")
        coverage_valid = args.validation_worker or (
            set(counts) == set(OWNED)
            and all(
                c["num_statements"] > 0 and c["covered_lines"] == c["num_statements"]
                for c in counts.values()
            )
        )
        failed = any(r["required"] and not r["passed"] for r in receipts) or not coverage_valid
        if failed:
            value.update(
                honest_verdict="complete_disqualified_owned_checks",
                verdict_class="disqualified",
                hardware_evidence_ready_score=0,
            )
            value["gate_check_summary"].append(
                h.operand(
                    "owned_validation",
                    raw / "validation_manifest.json",
                    "required_checks_and_coverage",
                    True,
                    False,
                    sha256_file(raw / "validation_manifest.json"),
                )
            )
            if not any(r["required"] and not r["passed"] for r in receipts):
                receipts.append(dict(name="nonempty_changed_coverage", required=True, passed=False))
        for ref in refs:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("frozen_code_drift")
        atomic_json(raw / "primitive_rows.json", dict(rows=value["rows"]))
        value.update(
            experiment_id=8003,
            task_id=TASK,
            milestone="2026.10.693",
            run_date=args.date,
            execution_date=args.date,
            invocation_timestamp=datetime.now(UTC).isoformat(),
            schema="carnot.hardware_sparse_boundary.v693.v1",
            inference_substrate="verifier_ensemble_against_cached_candidates"
            if plan["sparse_available"]
            else "aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_specs=[],
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            random_seed=69303,
            verifier_is_oracle=False,
            claim_scope="CPU fixed-point emulation of a fitted sparse head. Historical board custody remains dated. No FPGA, NPU, TSU execution or device speed claim. Natural source annotations are fallible; numerical agreement is not scientific benefit.",
            methodology="Freeze signed Q12 formats, fit-only LUT geometry, saturated integer products and sums, nearest-even rounding, one sparse update and host final readout. At most 128 fit/tune source groups. Knot and saturation fixtures are protocol controls. Measured duration is not padded.",
            methodology_note="Exact action agreement is deterministic parity on cached inputs, not perfect natural-data classification. Seeds and formats never multiply independent sources.",
            external_design_references=[
                dict(
                    url="https://extropic.ai/writing/z1t",
                    scope="external_design_no_local_device_evidence",
                ),
                dict(
                    url="https://arxiv.org/abs/2602.02056",
                    title="Ultrafast On-chip Online Learning via Spline Locality in Kolmogorov-Arnold Networks",
                    scope="external_spline_design_no_local_accelerator_result",
                ),
            ],
            replay_inputs=plan,
            replay_input_reference=dict(
                path=str(raw / "replay_inputs.json"), sha256=sha256_file(raw / "replay_inputs.json")
            ),
            config=CONFIG,
            validation_command_manifest_path=str(raw / "validation_manifest.json"),
            validation_receipts=receipts,
            coverage_statement_counts=counts,
            code_config_hashes=refs,
            raw_shard_hashes=[
                dict(
                    path=str(raw / "primitive_rows.json"),
                    sha256=sha256_file(raw / "primitive_rows.json"),
                )
            ],
            repository_health=[r for r in receipts if not r["required"]],
            scratch_root_receipt=dict(
                path=str(scratch),
                outside_checkout=True,
                retained_for_replay=True,
                artifact_guard_enabled=True,
            ),
            flagged_adversarial=False,
            validation_worker=args.validation_worker,
        )
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=CONFIG, inputs=plan, code=refs, seed=69303, rows=value["rows"])
        )
        phase("freeze_candidate")
        value["duration_s"] = time.monotonic() - began
        spans[-1]["end_s"] = value["duration_s"]
        value["phase_spans"] = spans
        value["duration_scope"] = (
            "Monotonic invocation through candidate freeze; terminal validator timing is in its hash-bound sidecar."
        )
        publish(args.output.absolute(), value, scratch)
        phase("published_checked_primary")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[exp8003] terminal_error={error}", flush=True)
        return 1
