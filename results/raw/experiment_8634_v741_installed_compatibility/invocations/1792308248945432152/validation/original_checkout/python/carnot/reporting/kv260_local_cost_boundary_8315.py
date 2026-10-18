"""REQ-REPORT-8315: audit cost operands before spending benchmark work.

This invocation has disqualified fixture inputs and no natural primary. Its
terminal record preserves those external blocks without manufacturing timings.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.request_trace_inventory_8200 import operand

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8315_v717_kv260_local_cost_boundary"
CLI = "scripts/experiments/" + NAME + ".py"
MODULE = "python/carnot/reporting/kv260_local_cost_boundary_8315.py"
RUNNER = "python/carnot/reporting/kv260_local_cost_execution_8315.py"
TEST = "tests/python/test_kv260_local_cost_boundary_8315.py"
OWNED = [MODULE, RUNNER, CLI]
MODEL_SPECS: list[Json] = []
SOURCES = dict(
    fixture=("experiment_8306_v717_local_update_isolation", "local_kernel_ready_score"),
    natural=("experiment_8310_v717_continuous_local_learning", "trajectory_ready_score"),
    hardware=("experiment_8301_v716_kv260_evidence_cost_boundary", None),
)
OPERATIONS = [
    "basis_construction",
    "index_lookup",
    "index_invalidation",
    "all_cache_invalidation",
    "gradient_update",
    "prediction",
    "serialization",
    "fsync_acknowledgment",
    "restart",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each boundary so child supervision can distinguish progress from a stall."""
    print(f"[exp8315] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Keep exact file bytes accountable rather than trusting a parsed JSON label."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def pin(path: Path, raw: Path, refs: list[Json]) -> Path:
    """Copy authenticated evidence so a later producer cannot alter this invocation."""
    digest = sha256_file(path)
    target = raw / "inputs" / (digest[7:] + ".bin")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    if sha256_file(target) != digest:
        raise ValueError("source_changed_during_copy")
    refs.append(dict(reference(target), original_path=str(path.absolute())))
    return target


def probe(path: Path, raw: Path, work: Json, field: str | None) -> Json | None:
    """Separate byte authentication from scientific eligibility, including failed checks."""

    def gate(location: Path, key: str, expected: Any, observed: Any) -> None:
        work["checks"].append(operand(key, location, expected, observed))

    first_check = len(work["checks"])
    gate(path, "exists", True, path.is_file())
    if not path.is_file():
        return None
    try:
        value: Json = json.loads(pin(path, raw, work["refs"]).read_bytes())
        validate_primary(value, path)
        terminal = Path(value["terminal_validation_sidecar_path"])
        receipt = json.loads(pin(terminal, raw, work["refs"]).read_bytes())
        publication = receipt["publication"]
        digest = sha256_file(path)
        gate(terminal, "publication.primary_sha256", digest, publication.get("primary_sha256"))
        gate(
            terminal,
            "publication.primary_path",
            str(path.absolute()),
            publication.get("primary_path"),
        )
        sidecar = Path(publication["sidecar_path"])
        report = read_bound_sidecar(path, sidecar)
        pin(sidecar, raw, work["refs"])
        gate(sidecar, "primary_path", str(path.absolute()), report.get("primary_path"))
        gate(sidecar, "report.passed", True, report.get("report", {}).get("passed"))
        for key, expected in [("required_checks_passed", True), ("flagged_adversarial", False)]:
            gate(path, key, expected, value.get(key))
        if field is not None:
            gate(path, field, 1, value.get(field))
        for ref in value.get("raw_shard_hashes", []):
            location = Path(ref["path"])
            gate(
                location,
                "sha256",
                ref["sha256"],
                sha256_file(location) if location.is_file() else None,
            )
            pin(checked(ref), raw, work["refs"])
        gate(path, "authentication", "valid", "valid")
        return value if all(row["passed"] for row in work["checks"][first_check:]) else None
    except (OSError, ValueError, KeyError, TypeError) as error:
        gate(path, "authentication", "valid", str(error))
        return None


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Check external operands in private scratch; unqualified data is never timed."""
    began = time.monotonic()
    work: Json = dict(
        checks=[],
        refs=[],
        hardware={},
        hardware_transcript={},
        fixture=fixture,
        operand_root=str(root.absolute()),
    )
    with TemporaryDirectory(prefix="carnot-8315-preconditions-") as directory:
        private = Path(directory)
        (private / "write_probe").write_bytes(b"private")
        work["checks"].append(operand("private_scratch", private, True, True))
        for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
            path = ROOT / ".venv/bin" / name
            work["checks"].append(operand("tool_available", path, True, path.is_file()))
        work["checks"].append(
            operand(
                "available_disk_at_least_1GiB",
                private,
                True,
                shutil.disk_usage(private).free >= 1024**3,
            )
        )
    for index, (branch, (name, field)) in enumerate(SOURCES.items()):
        progress("before_authenticate_" + branch, index, len(SOURCES) - index)
        value = probe(root / "results" / (name + ".json"), raw, work, field)
        if branch == "hardware" and value is not None:
            work["hardware"] = value
            historical = value.get("kv260_obligation", {}).get("historical", {})
            if historical.get("source_transcript"):
                transcript = Path(historical["source_transcript"])
                expected = historical["source_transcript_sha256"]
                observed = sha256_file(transcript) if transcript.is_file() else None
                work["checks"].append(operand("sha256", transcript, expected, observed))
                if expected == observed:
                    work["hardware_transcript"] = json.loads(
                        pin(transcript, raw, work["refs"]).read_bytes()
                    )
                else:
                    work["hardware"] = {}
        progress("after_authenticate_" + branch, index + 1, len(SOURCES) - index - 1)
    work["duration_s"] = time.monotonic() - began
    work["phase_spans"] = [dict(phase="cost_operand_preconditions", duration_s=work["duration_s"])]
    work["code_config_hashes"] = [
        reference(ROOT / p)
        for p in [
            *OWNED,
            TEST,
            "python/carnot/reporting/primary_publication.py",
            "ops/exclusion_manifest.yaml",
            "python/carnot/reporting/local_update_execution_8306.py",
            "python/carnot/reporting/methods_stream_execution_8111.py",
            "python/carnot/verify/hard_exit_learning_qualification_8206.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    ]
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Unavailable operands remain censored rows; owned failures cannot gain readiness."""
    failed = [row for row in work["checks"] if not row["passed"]]
    if not failed:
        raise ValueError("qualified_operands_require_complete_cost_measurement")
    owned = bool(receipts) and all(row["passed"] for row in receipts)
    klass = "blocked" if owned else "disqualified"
    rows = [
        dict(
            source_id=branch,
            condition=branch + ":" + arm,
            arm=arm,
            intended=1,
            completed=False,
            failed=False,
            censored=True,
            excluded=False,
            independent=0,
            numerator=0,
            denominator=1,
            eligible=False,
            cost_ns=None,
            evidence_status="unavailable_external_operand",
        )
        for branch in ["fixture", "natural"]
        for arm in ["full", "indexed"]
    ]
    value: Json = dict(
        experiment_id=8315,
        task_id="exp8315-kv260-local-cost-boundary",
        milestone="2026.10.717",
        run_date="20261008",
        honest_verdict="complete_"
        + klass
        + "_"
        + (failed[0]["upstream"] if owned else "owned_checks"),
        verdict_class=klass,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["hardware"].get("historical_model_provenance", {}),
        rows=rows,
        intended_count=4,
        completed_count=0,
        failed_count=0,
        censored_count=4,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            intended_branch_arm_rows=4,
            fixture_trajectory_budget=48,
            natural_source_count=None,
            repetitions_per_arm=5,
            repetitions_are_independent=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_checks=owned,
            qualified_operands=False,
            complete_transaction_measurement=False,
            state_action_parity=None,
            natural_paired_median_ratio_at_most_0_9=None,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str((raw / "terminal_validation.json").absolute()),
        preconditions_checked=True,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178315,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[reference(raw / "measurement.json")],
        measurement_reference=reference(raw / "measurement.json"),
        cited_upstream_artifacts=[
            dict(
                branch=branch,
                path=str(Path(work["operand_root"]) / "results" / (name + ".json")),
                sha256=next(
                    (
                        r["sha256"]
                        for r in work["refs"]
                        if Path(r["original_path"]).name == name + ".json"
                    ),
                    None,
                ),
                imported_fields=[field] if field else ["kv260_obligation"],
            )
            for branch, (name, field) in SOURCES.items()
        ],
        cpu_cost_ready_score=0,
        natural_cost_ready_score=0,
        kv260_boundary_ready_score=0,
        complete_cost_rows=[],
        fixed_point_rows=[],
        operation_rows=[
            dict(
                operation=name,
                assigned_substrate="CPU",
                measured_ns=None,
                kv260_supported=False,
                status="unavailable_cost",
                reason="deployed_quadratic_Ising_overlay_has_no_spline_or_durability_kernel",
            )
            for name in OPERATIONS
        ],
        kv260_obligation=work["hardware"].get("kv260_obligation", {}),
        historical_overlay=work["hardware_transcript"].get("kv260_overlay_loaded"),
        synthesis_status=work["hardware_transcript"].get(
            "synthesis_status", "unavailable_in_authenticated_transcript"
        ),
        current_device_execution_count=0,
        compatible_fraction=None,
        ideal_service_speedup_bound=None,
        live_acquisition_cost_status="unavailable",
        fixed_point_status="unavailable_qualified_update_operands",
        fixed_point_protocol=dict(
            coefficient_bits=16,
            fractional_bits=12,
            arithmetic="saturating",
            reference="float64",
            decision_boundaries=[0.25, 0.75],
            hardware_execution=False,
        ),
        invocation_argv=work.get("invocation_argv", []),
        fixture=work["fixture"],
        nfr01_met=False,
        nfr01_status="unavailable_complete_service_costs",
    )
    value["field_principles"] = {
        key: "Preserve the current invocation and exact external block; unavailable evidence is not measured zero."
        for key in value
    }
    value["field_principles"].update(
        complete_cost_rows="Only qualified, paired complete transactions can supply timing rows; no costs imported from disqualified primaries.",
        operation_rows="CPU assignment describes unsupported overlay operations; absent clocks cannot establish compatible fractions.",
        fixed_point_rows="Numerical emulation requires qualified coefficients; empty rows claim no software or board result.",
        kv260_obligation="Preserve the authenticated historical transcript and k<=5 boundary without rerunning board availability.",
        reproducibility_checksum="Bind the reduction to its primitive inputs and validation receipts.",
        field_principles="Explain each field's evidentiary purpose without replacing values with prose.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild the entire terminal reduction; rehashing a changed summary cannot pass."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            checked(ref)
        work = json.loads(checked(value["measurement_reference"]).read_bytes())
        for receipt in value["validation_receipts"]:
            for prefix in ["stdout", "stderr"]:
                if receipt.get(prefix + "_path"):
                    checked(
                        dict(path=receipt[prefix + "_path"], sha256=receipt[prefix + "_sha256"])
                    )
        return bool(
            build(
                work,
                Path(value["measurement_reference"]["path"]).parent,
                value["validation_receipts"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
