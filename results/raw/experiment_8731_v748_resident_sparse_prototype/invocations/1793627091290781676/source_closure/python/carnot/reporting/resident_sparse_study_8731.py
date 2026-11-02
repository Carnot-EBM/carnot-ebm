"""REQ-VERIFY-8731: configure existing custody for the resident prototype."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import compiled_sparse_runner_8719 as old_runner
from carnot.reporting import compiled_sparse_study_8719 as old_study
from carnot.reporting import v748_collection_contract as current
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import resident_sparse_8731 as n

Json = dict[str, Any]
e = current.namespace(old_study, "resident_study_8731")
runtime = current.namespace(old_runner, "resident_runner_8731")
e.NAME, e.TASK = (
    "experiment_8731_v748_resident_sparse_prototype",
    "exp8731-resident-sparse-prototype",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.OWNED = [
    "python/carnot/verify/resident_sparse_8731.py",
    "python/carnot/reporting/resident_sparse_study_8731.py",
    e.CLI,
]
e.TESTS = [
    "tests/python/test_resident_sparse_8731.py",
    "tests/python/test_resident_sparse_runner_8731.py",
]
e.RUST += [
    "crates/carnot-python/src/spline_resident_8731.rs",
    "crates/carnot-python/src/spline_resident_8731_tests.rs",
]
e.contract = SimpleNamespace(authority=current.m.authority)
e.numerical, e.progress = n, n.progress
runtime.e = e
base_inputs, base_build, base_native_build = e.inputs, e.build, runtime.native_build
PRODUCER = "results/experiment_8719_v747_compiled_sparse_update.json"
PIN = "sha256:9d27b1f8ca99b50d0452d7e1d204c18223a16b7eec36208d8bc5c44830c6f1ac"


def inputs() -> Json:
    """Authenticate retained native witnesses without rerunning their producer."""
    value: Json = base_inputs()
    ref = dict(path=str(e.ROOT / PRODUCER), sha256=PIN)
    value["refs"].append(ref)
    value["refs"] += [
        e.reference(e.ROOT / p)
        for p in [
            "python/carnot/reporting/v748_collection_contract.py",
            "python/carnot/reporting/roadmap_contract.py",
            "python/carnot/reporting/compiled_sparse_runner_8719.py",
            "python/carnot/reporting/compiled_sparse_study_8719.py",
            "python/carnot/reporting/spline_lifecycle_study_8706.py",
            "python/carnot/verify/compiled_sparse_8719.py",
        ]
    ]
    operand = ref
    try:
        producer = e.read(ref)
        for field, expected in [
            ("experiment_id", 8719),
            ("native_update_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            row = dict(
                e.previous.gate(ref, field, expected), upstream="exp8719-compiled-sparse-update"
            )
            if not row["passed"]:
                value["failures"].append(row)
        value["refs"] += e.previous.terminal(ref, producer)
        operand = producer["primitive_reference"]
        work = e.read(operand)
        operand = producer["validation_reference"]
        checks = e.read(operand)
        if (
            work["private_control"]
            or not all(r["passed"] for r in checks["receipts"])
            or checks["receipts"] != producer["validation_receipts"]
        ):
            raise ValueError("qualified_native_primitives")
        for source in work["code_refs"]:
            relative = Path(source["path"]).relative_to(e.ROOT)
            snapshot = dict(
                path=str(Path(work["raw"]) / "source_closure" / relative), sha256=source["sha256"]
            )
            operand = snapshot
            e.verify(operand)
            value["refs"].append(snapshot)
        for source in work["raw_refs"] + [
            producer["loaded_extension_reference"],
            producer["primitive_reference"],
            producer["validation_reference"],
        ]:
            operand = source
            e.verify(operand)
            value["refs"].append(source)
        if len(work["summaries"]) != 25:
            raise ValueError("original_native_job_count")
        for job, summary in zip(value["jobs"], work["summaries"], strict=True):
            if e.read(summary["job_reference"]) != job or not summary["passed"]:
                raise ValueError("original_native_job_identity")
        value["historical_refs"] = [s["rows_reference"] for s in work["summaries"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        value["failures"].append(
            dict(
                upstream="exp8719-compiled-sparse-update",
                **operand,
                check="qualified_native_operands",
                field="original_native_witnesses",
                operator="==",
                expected="authenticated",
                observed=str(error),
                operand_state="missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand",
                passed=False,
            )
        )
    return value


def native_build(raw: Path) -> list[Json]:
    """Enable the resident feature in both the real crate and scoped test harness."""
    real_child = runtime.child

    def build_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        """The shared build procedure retains genuine exits and both output streams."""
        if name == "prepare_harness_format":
            manifest = raw / "native_checks/Cargo.toml"
            with manifest.open("a") as stream:
                stream.write(
                    '\n[features]\ndefault=["spline-resident-8731"]\nspline-resident-8731=[]\n'
                )
        if name == "cargo_build":
            argv = ["spline-resident-8731" if x == "spline-fixed-8719" else x for x in argv]
        return dict(real_child(name, argv, logs, **kwargs))

    with (
        patch.object(runtime, "child", build_child),
        patch.dict(
            runtime.os.environ, LLVM_PROFILE_FILE=str(raw / "native_checks/native-%p-%m.profraw")
        ),
    ):
        receipts = list(base_native_build(raw))
    if not all(row["passed"] for row in receipts):
        return receipts
    project = raw / "native_checks"
    binary = next(
        p
        for p in (project / "target/debug/deps").glob("spline_check_8719-*")
        if p.is_file() and not p.suffix
    )
    profile = project / "native.profdata"
    receipts.append(
        real_child(
            "rust_coverage_merge",
            [
                "llvm-profdata",
                "merge",
                "-sparse",
                *[str(p) for p in sorted(project.glob("native-*.profraw"))],
                "-o",
                str(profile),
            ],
            raw / "build_logs",
        )
    )
    receipt = real_child(
        "rust_coverage_export",
        [
            "llvm-cov",
            "export",
            str(binary),
            "--instr-profile=" + str(profile),
            str(e.ROOT / "crates/carnot-python/src/spline_resident_8731.rs"),
        ],
        raw / "build_logs",
    )
    receipts.append(receipt)
    summary = json.loads(Path(receipt["stdout_path"]).read_bytes())["data"][0]["files"][0][
        "summary"
    ]
    receipts.append(
        dict(
            name="rust_owned_coverage",
            passed=summary["lines"]["percent"] == 100 and summary["regions"]["percent"] == 100,
            coverage=summary,
            references=[e.reference(profile), e.reference(Path(receipt["stdout_path"]))],
        )
    )
    return receipts


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Only exact numerical, restart, loaded-binary and owned-check gates release timing."""
    value: Json = base_build(work, receipts, output)
    bindings = [
        dict(head_index=s["head_index"], sequence=s["sequence"], **row)
        for s in work["summaries"]
        for row in s["binding_call_rows"]
    ]
    value.update(
        experiment_id=8731,
        milestone="2026.11.748",
        honest_verdict=value["honest_verdict"].replace(
            "compiled_sparse_update", "resident_sparse_prototype"
        ),
        inference_substrate="deterministic_verifier",
        binding_call_rows=bindings,
        transferred_byte_rows=bindings,
        resident_update_ready_score=value["native_update_ready_score"],
        byte_accounting="Logical typed field bytes; Python allocator overhead is unmeasured. Qualification telemetry and coefficient copies are included separately; no performance measurement.",
        serialization_policy="Explicit host checkpoints; durable writes, fsync and acknowledgements remain host-owned.",
        speedup_claim=None,
    )
    value["branch_gate_values"] = dict(
        resident_update_ready_score=value["resident_update_ready_score"]
    )
    value["field_principles"].update(
        {
            k: "Bind eligibility to exact sequential arithmetic, resident counters and durable restart; no learning or performance claim."
            for k in value
        }
    )
    return value


e.inputs, e.build, runtime.native_build = inputs, build, native_build
main = runtime.main
