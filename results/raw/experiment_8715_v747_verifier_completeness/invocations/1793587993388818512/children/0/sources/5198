"""REQ-REPORT-8593: identify numerical inference without claiming language-model work.

Original producer seals define the head. A repaired report checksum cannot
substitute a different checkpoint, scaler, implementation or decision policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import resident_complete_cost_8584 as resident
from carnot.reporting import nine_input_inputs_8533 as inputs
from carnot.reporting import v738_contract_methods as methods
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, progress
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify.evidence_features_7980 import FEATURES
from scripts.adversarial_verify import _classify_current_task_inference_claim, verify_artifact

Json = dict[str, Any]
ROOT = resident.ROOT
NAME = "experiment_8593_v738_resident_provenance_qualification"
TASK = "exp8593-resident-provenance-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_resident_provenance_8593.py"
OWNED = [
    f"python/carnot/reporting/resident_{s}_8593.py" for s in ["provenance", "runner", "validation"]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum = resident.reference, resident.read, resident.checksum
HISTORICAL = "results/experiment_8584_v737_resident_complete_cost.json"


def original(root: Path = ROOT) -> Json:
    """Keep historical bytes separate from every private correction."""
    return dict(read(dict(path=str(root / HISTORICAL), sha256=HISTORICAL_PIN)))


def descriptor(root: Path) -> Json:
    """Derive identity from the original fitted-head seal, never from candidate labels."""
    fitted = read(dict(path=str(root / inputs.HEAD), sha256=inputs.HEAD_PIN))
    checkpoint = dict(path=fitted["frozen_heads_path"], sha256=fitted["frozen_heads_sha256"])
    head = read(checkpoint)["heads"]["spline9"][0]
    bundle = resident.admit(root)
    resident.require(root, "original_checkpoint_head", head, bundle["head"])
    value = dict(
        name="Exp8431 spline9 seed17 calibrated energy head",
        original_producer=8431,
        checkpoint=checkpoint,
        checkpoint_head="heads.spline9[0]",
        seed=head["seed"],
        parameters_sha256=canonical_hash(head["parameters"]),
        parameter_count=109,
        scaler_sha256=canonical_hash(head["scaler"]),
        temperature=head["temperature"],
        temperature_sha256=canonical_hash(head["temperature"]),
        decay_scale=head["decay_scale"],
        basis=dict(degree=3, knots=sparse.KNOTS, coefficients_per_input=12, bias=1),
        dtype="IEEE754 binary64",
        feature_order=["q_raw_p2", *FEATURES],
        native_extension_build=bundle["extension"],
        policy=dict(
            name="v726_decision_metrics.action",
            thresholds=[0.2, 0.8],
            source=reference(ROOT / "python/carnot/verify/v726_decision_metrics.py"),
        ),
        normalization="raw-p2 logit offset; temperature; stable two-energy normalization; native threshold fallback",
    )
    return dict(value, descriptor_sha256=canonical_hash(value))


def check_descriptor(value: Any) -> bool:
    """Exact equality rejects plausible descriptions that name a different real head."""
    return bool(value == descriptor(ROOT))


def reproduce(raw: Path, target: Json) -> Json:
    """Run the unchanged classifier and CLI on original and metadata-only corrected bytes."""
    result = {}
    for label in ["original", "corrected"]:
        value = original()
        if label == "corrected":
            value["target_model"] = target
            value["reproducibility_checksum"] = checksum(value)
        directory = raw / label
        path = directory / Path(HISTORICAL).name
        atomic_json(path, value)
        scan = verify_artifact(path)
        receipt = child(
            label + "_classifier",
            [sys.executable, "-u", "scripts/adversarial_verify.py", "--json", str(path)],
            directory / "logs",
            expected=1 if label == "original" else 0,
            deadline=60,
        )
        refused = False
        try:
            publication = publish_primary(
                directory / "publication" / path.name,
                value,
                lambda p: dict(passed=not verify_artifact(p)["flags"]),
            )
        except ValueError:
            refused, publication = True, None
        result[label] = dict(
            candidate=reference(path),
            classifier_inputs={
                k: value.get(k)
                for k in [
                    "inference_substrate",
                    "inference_substrate_class",
                    "MODEL_SPECS",
                    "model_invocation_counts",
                    "historical_model_provenance",
                    "target_model",
                    "random_seed",
                    "duration_s",
                    "reproducibility_checksum",
                ]
            },
            classifier_output=_classify_current_task_inference_claim(value),
            flags=scan["flags"],
            cli=receipt,
            publication_refused=refused,
            publication=publication,
        )
    return result


def measure(raw: Path, run_date: str, *, root: Path = ROOT) -> Json:
    """Authenticate independent CPU resources before reproducing the provenance defect."""
    began = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        raw=str(raw),
        run_date=run_date,
        started_ns=began,
        gate_check_summary=[],
        target_model=None,
        reproduction={},
        sources=[],
        preconditions_checked={},
    )
    try:
        with TemporaryDirectory(prefix="exp8593-", dir="/var/tmp") as scratch:
            probe = Path(scratch) / "writable"
            probe.write_bytes(b"disk-backed scratch")
            mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
            filesystem = max(
                (row for row in mounts if Path(scratch).is_relative_to(row[1])),
                key=lambda row: len(row[1]),
            )[2]
            resident.require(
                Path(scratch), "disk_backed_filesystem", True, filesystem not in {"tmpfs", "ramfs"}
            )
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            work["preconditions_checked"] = dict(
                scratch_parent="/var/tmp",
                scratch_filesystem=filesystem,
                scratch_writable=probe.read_bytes() == b"disk-backed scratch",
                available_memory_bytes=available,
                disk_free_bytes=shutil.disk_usage(scratch).free,
                measurement_limit_s=3600,
                validation_reserve_s=900,
                hard_cap_s=4800,
                optional_devices_required=False,
                tools=[
                    reference(ROOT / ".venv/bin" / name)
                    for name in ["python", "pytest", "ruff", "mypy", "coverage"]
                ],
            )
            resident.require(Path(scratch), "memory_available", True, available > 64 * 1024**2)
            resident.require(
                Path(scratch),
                "disk_available",
                True,
                shutil.disk_usage(scratch).free > 64 * 1024**2,
            )
        authority = methods.authority(root)
        resident.require(root, "eleven_task_authority", True, authority["ready"])
        work["authority"] = authority
        work["target_model"] = descriptor(root)
        old = original(root)
        bundle = resident.admit(root)
        work["sources"] = (
            bundle["refs"]
            + authority["references"]
            + [
                reference(root / HISTORICAL),
                reference(root / inputs.HEAD),
                work["target_model"]["checkpoint"],
                reference(ROOT / methods.PROTOCOL),
                *[
                    reference(ROOT / name)
                    for name in [
                        "scripts/adversarial_verify.py",
                        "scripts/verdict_row_consistency_lint.py",
                        "python/carnot/reporting/primary_publication.py",
                        "scripts/conductor_gates.py",
                        "scripts/in_process_doc_reconcile.py",
                    ]
                ],
            ]
        )
        work["historical_model_provenance"] = old["historical_model_provenance"]
        work["preserved_quarantines"] = old["preserved_quarantines"]
        work["original_node_union"] = old["exact_union_receipt"]
        progress("8593_before_classifier_reproduction")
        work["reproduction"] = reproduce(raw / "reproduction", work["target_model"])
        progress("8593_after_classifier_reproduction", 2, 0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["gate_check_summary"].append(
            error.row
            if isinstance(error, resident.GateFailure)
            else resident.GateFailure(root, "original_resources", True, str(error)).row
        )
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    return work


HISTORICAL_PIN = "sha256:356aad75439ecaedbbe102e372dc28c2b92e15c848e0d6099165adb6a53995ed"
REQUIRED = {
    "exact_disjoint_node_union",
    "owned_python_coverage_100",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "private_publication",
    "cold",
    "missing",
    "owned_failure",
    "rehashed_tamper",
    "absent_descriptor",
    "wrong_head",
    "inconsistent_scaler",
}


def receipts_valid(work: Json, receipts: list[Json]) -> bool:
    """Derived publication claims need the actual accepted bytes and child logs."""
    ordinary = [r for r in receipts if r["name"] != "private_publication"]
    private = work.get("private_publication_receipts", [])
    return bool(
        resident.receipt_bytes(work, ordinary)
        and private
        and all(
            r["validators"]["passed"]
            and r["consumer"]["passed"]
            and resident.receipt_bytes({}, [*r["validators"]["receipts"], r["consumer"]])
            and resident.read_bound_sidecar(
                Path(r["publication"]["primary_path"]), Path(r["publication"]["sidecar_path"])
            )["report"]["passed"]
            for r in private
        )
    )


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Only earned consumer and validation receipts establish measurement readiness."""
    checks = {r["name"]: r["passed"] for r in receipts}
    required = REQUIRED <= checks.keys() and all(checks.values()) and receipts_valid(work, receipts)
    reproduction = work["reproduction"]
    provenance = bool(
        reproduction
        and reproduction["original"]["publication_refused"]
        and reproduction["original"]["cli"]["passed"]
        and reproduction["corrected"]["cli"]["passed"]
        and not reproduction["corrected"]["flags"]
        and check_descriptor(work["target_model"])
    )
    ready = required and provenance and not work["gate_check_summary"]
    verdict = (
        "disqualified"
        if any(not r["passed"] for r in receipts)
        else "blocked"
        if work["gate_check_summary"]
        else "null"
        if ready
        else "disqualified"
    )
    rows = [
        dict(
            unit_id=label,
            passed=r["cli"]["passed"],
            status="completed" if r["cli"]["passed"] else "failed",
            intended_count=1,
            completed_count=int(r["cli"]["passed"]),
            failed_count=int(not r["cli"]["passed"]),
            censored_count=0,
            missing_reason=None if r["cli"]["passed"] else "classifier_exit",
        )
        for label, r in sorted(reproduction.items())
    ]
    gates = dict(
        original_resources=not work["gate_check_summary"],
        truthful_provenance=provenance,
        required_validation=bool(required),
    )
    value = dict(
        experiment_id=8593,
        task_id=TASK,
        milestone="2026.10.738",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v738_resident_provenance_qualification",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"]
        + [
            resident.GateFailure(
                Path(str(r.get("stdout_path", work["raw"]))), r["name"], True, False
            ).row
            for r in receipts
            if not r["passed"]
        ],
        inference_substrate="CPU complete resident and cold process measurements",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_llm_invocations=0,
        historical_model_provenance=work.get("historical_model_provenance", []),
        target_model=work["target_model"],
        head_descriptor_reference=reference(Path(work["raw"]) / "head_descriptor.json")
        if work["target_model"]
        else None,
        rows=rows,
        intended_count=2,
        completed_count=sum(r["completed_count"] for r in rows),
        failed_count=sum(r["failed_count"] for r in rows),
        censored_count=2 - len(rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            provenance_candidates=2, independent_scientific_units=0, timing_republished=False
        ),
        verifier_is_oracle=True,
        exposure_scope="original exposed head; provenance qualification only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(required),
        flagged_adversarial=verdict == "disqualified",
        acceptance_gates=gates,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions_checked"],
        duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="resource_and_provenance_qualification",
                start_ns=work["started_ns"],
                end_ns=work["ended_ns"],
            )
        ],
        random_seed=7368573,
        source_artifact_hashes=work["sources"],
        code_config_hashes=[reference(ROOT / p) for p in OWNED + [TEST]],
        raw_shard_hashes=[reference(Path(work["raw"]) / "work.json")],
        cited_upstream_artifacts=work["sources"],
        resident_measurement_ready_score=int(ready),
        classifier_reproduction_rows=reproduction,
        descriptor_mutation_rows=work.get("descriptor_mutation_rows", []),
        original_node_union=work.get("original_node_union"),
        exact_union_receipt=work.get("exact_union_receipt"),
        python_coverage=work.get("python_coverage"),
        private_publication_receipts=work.get("private_publication_receipts", []),
        current_source_reference=reference(ROOT / methods.PROTOCOL),
        work_reference=reference(Path(work["raw"]) / "work.json"),
        publication_output=str(output),
        preserved_quarantines=work.get("preserved_quarantines", []),
        historical_cost_qualification="Exp8584 remains disqualified. Exp8559 cold-cost null, Exp8529 source null and Exp8553 label rejection remain unchanged. Fresh timing belongs to Exp8596.",
        methodology_note="No language model is loaded. The original small numerical energy head is evaluated by unchanged resident tests. Qualification earns no speed or learning credit.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to immutable original evidence; qualification supplies no scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        model_invocation_counts="These counters cover current language-model work only. The unchanged resident tests separately evaluate the numerical energy head.",
        target_model="Exact original checkpoint, scaler, temperature, basis, dtype, extension build, feature order and policy identify the numerical head, not an LLM.",
        resident_measurement_ready_score="One requires truthful provenance, complete original resident test union and exact unchanged private publication acceptance; old costs remain quarantined.",
        classifier_reproduction_rows="Record exact original and metadata-only classifier inputs, outputs, warning exits and publication refusal.",
        original_node_union="Historical node identities and authority are preserved separately from this invocation's disjoint execution union.",
        descriptor_mutation_rows="Genuine consumer children reject absent, wrong-head and repaired-hash inconsistent descriptors.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild from immutable primitives and original seals, not repaired summary hashes."""
    try:
        value = json.loads(path.read_bytes())
        if value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        for ref in value["source_artifact_hashes"] + value["code_config_hashes"]:
            read_bytes = Path(ref["path"])
            resident.require(read_bytes, "immutable_bytes", ref["sha256"], sha256_file(read_bytes))
        if work["target_model"] is not None:
            if (
                not check_descriptor(value["target_model"])
                or work["target_model"] != value["target_model"]
            ):
                return False
            resident.require(path, "original_authority", methods.authority(ROOT), work["authority"])
            for label, row in work["reproduction"].items():
                candidate = read(row["candidate"])
                expected = original()
                if label == "corrected":
                    expected["target_model"] = descriptor(ROOT)
                    expected["reproducibility_checksum"] = checksum(expected)
                resident.require(path, "metadata_only_private_candidate", expected, candidate)
                resident.require(
                    path,
                    "classifier_flags",
                    verify_artifact(Path(row["candidate"]["path"]))["flags"],
                    row["flags"],
                )
                resident.require(
                    path,
                    "classifier_output",
                    _classify_current_task_inference_claim(candidate),
                    row["classifier_output"],
                )
                resident.require(
                    path, "classifier_logs", True, resident.receipt_bytes({}, [row["cli"]])
                )
            resident.require(
                path,
                "head_descriptor_bytes",
                value["target_model"],
                read(value["head_descriptor_reference"]),
            )
        return bool(
            value == artifact(work, value["validation_receipts"], Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def consume(path: Path, *, require_ready: bool = True) -> Json:
    """A downstream reader refuses unearned readiness and mismatched numerical identity."""
    value = json.loads(path.read_bytes())
    if not replay(path) or (require_ready and value["resident_measurement_ready_score"] != 1):
        raise ValueError("resident_provenance_not_ready")
    return dict(
        passed=True,
        path=str(path),
        sha256=reference(path)["sha256"],
        resident_measurement_ready_score=value["resident_measurement_ready_score"],
    )
