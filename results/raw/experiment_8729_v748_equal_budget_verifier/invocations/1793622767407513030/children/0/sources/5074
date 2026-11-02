"""REQ-REPORT-8558: immutable versions bind candidates to their actual operands."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import nine_input_qualification_8545 as previous
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
__all__ = ["previous"]
legacy = previous.legacy
inputs = previous.inputs
ROOT = previous.ROOT
NAME = "experiment_8558_v735_nine_input_native_qualification"
TASK = "exp8558-nine-input-native-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_nine_input_8558.py"
OWNED = [
    f"python/carnot/reporting/nine_input_{p}_8558.py"
    for p in ["qualification", "runner", "validation"]
] + [CLI]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8545_v734_nine_input_native_qualification.json"
UPSTREAM_PIN = "sha256:5c4cef207e9972a6ec5d70347918c186dba832f0d45776d2df35167e0548955a"
original_authority = previous.original_authority
REQUIRED = {
    "exact_disjoint_node_union",
    "owned_python_coverage_100",
    "owned_rust_coverage_100",
    "private_complete_cli",
    "cold",
    "missing",
    "owned_failure",
    "rehashed_tamper",
    "adversarial",
    "strict_rows",
    "original_owned_failures",
    "snapshot_order_reproduction",
    "terminal_cold",
}


def seal(work: Json, raw: Path, version: str) -> Json:
    """Copy before binding so later work cannot rewrite an earlier candidate's input."""
    result = deepcopy(work)
    folder = raw / "snapshots" / version
    folder.mkdir(parents=True, exist_ok=False)
    result["raw"] = str(folder.absolute())
    atomic_json(folder / "measurement.json", result)
    (folder / "measurement.json").chmod(0o444)
    return result


def measure(root: Path, raw: Path) -> Json:
    """Reuse authenticated compiled bytes while recomputing every numerical operand."""
    legacy.progress("start_8558_no_model_load_LLM_calls_0")
    upstream = inputs.read(dict(path=str(ROOT / UPSTREAM), sha256=UPSTREAM_PIN))
    extension = upstream["extension_reference"]
    if inputs.reference(Path(extension["path"]))["sha256"] != extension["sha256"]:
        raise ValueError("original_extension_hash")
    os.environ["CARNOT_8533_EXTENSION"] = extension["path"]
    (raw / "native_logs").mkdir(parents=True, exist_ok=True)
    os.environ["LLVM_PROFILE_FILE"] = str(raw / "native_logs/rust-%p-%m.profraw")
    work = previous.measure(root, raw)
    work["source_refs"].append(inputs.reference(ROOT / UPSTREAM))
    work["code_refs"].extend(inputs.reference(ROOT / p) for p in OWNED + [TEST])
    if work["extension"]:
        work["extension"]["receipts"] = extension["receipts"]
        if any(
            work["extension"][key] != extension[key]
            for key in ["sources", "compiler_flags", "sha256", "fast_math", "fma_contraction"]
        ):
            work["owned_failure"] = True
            work["failures"].append(
                legacy.diagnostic(
                    Path(extension["path"]),
                    "original_build_identity",
                    work["extension"],
                    expected=extension,
                )
            )
    work["historical_build_reference"] = inputs.reference(ROOT / UPSTREAM)
    return dict(work)


def reproduce(work: Json, raw: Path) -> tuple[Json, Json]:
    """Reconstruct the old write order using its sealed final work and real cold CLI."""
    upstream = inputs.read(dict(path=str(ROOT / UPSTREAM), sha256=UPSTREAM_PIN))
    historical = inputs.read(upstream["primitive_reference"])
    initial = deepcopy(historical)
    for key in [
        "original_failure_reproduction",
        "original_authority_by_suite",
        "exact_union_receipt",
        "coverage_config",
        "coverage_scratch",
        "python_coverage",
        "rust_coverage",
        "adversarial_findings",
        "total_duration_s",
    ]:
        initial.pop(key, None)
    initial["raw"] = str((raw / "historical_stale_snapshot").absolute())
    atomic_json(Path(initial["raw"]) / "measurement.json", initial)
    post = deepcopy(initial)
    for key in [
        "original_failure_reproduction",
        "original_authority_by_suite",
        "exact_union_receipt",
        "coverage_config",
        "coverage_scratch",
    ]:
        post[key] = historical[key]
    receipts = upstream["validation_receipts"][
        : next(i for i, r in enumerate(upstream["validation_receipts"]) if r["name"] == "cold")
    ]
    output = raw / (previous.NAME + ".json")
    expected = previous.artifact(post, receipts, output)
    reduced = previous.artifact(initial, receipts, output)
    candidate = raw / "historical_preliminary_candidate.json"
    atomic_json(candidate, expected)
    receipt = child(
        "snapshot_order_reproduction",
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / previous.CLI),
            "--cold-replay",
            str(candidate),
        ],
        raw / "reproduction",
        expected=1,
        deadline=120,
    )
    differences = [
        dict(field=k, expected=expected[k], reduced=reduced.get(k))
        for k in expected
        if expected[k] != reduced.get(k)
    ]
    diagnosis = dict(
        hypothesis_reproduced=receipt["passed"] and bool(differences),
        historical_cold_receipt=next(
            r for r in upstream["validation_receipts"] if r["name"] == "cold"
        ),
        reconstruction="Original final work with validation fields removed, then restored only in memory; initial bytes were not archived.",
        field_differences=differences,
        original_primary=inputs.reference(ROOT / UPSTREAM),
        snapshot_reference=inputs.reference(Path(initial["raw"]) / "measurement.json"),
    )
    receipt["passed"] = diagnosis["hypothesis_reproduced"]
    atomic_json(raw / "snapshot_order_diagnosis.json", diagnosis)
    return receipt, diagnosis


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness requires the same sealed work that the full reduction will read."""
    ref = inputs.reference(Path(work["raw"]) / "measurement.json")
    if inputs.read(ref) != work:
        raise ValueError("snapshot_drift")
    value = previous.artifact(work, receipts, output)
    checks = {r["name"]: r["passed"] for r in receipts}
    ready = (
        value["nine_input_native_ready_score"] == 1
        and REQUIRED <= checks.keys()
        and all(checks[n] for n in REQUIRED)
    )
    verdict = (
        "circular_positive"
        if ready
        else "blocked"
        if value["verdict_class"] == "blocked"
        else "disqualified"
    )
    value.update(
        experiment_id=8558,
        task_id=TASK,
        milestone="2026.10.735",
        honest_verdict=f"complete_{verdict}_v735_nine_input_native_qualification",
        verdict_class=verdict,
        nine_input_native_ready_score=int(ready),
        required_checks_passed=ready,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        primitive_reference=ref,
        preserved_quarantines=[8530, 8533, 8537, 8545],
        snapshot_order_diagnosis=work.get("snapshot_order_diagnosis", {}),
        historical_build_reference=work.get("historical_build_reference"),
    )
    value["acceptance_gates"]["all_required_contracts"] = ready
    value["gate_check_summary"] += [
        dict(
            check=r["name"],
            upstream=TASK,
            path=r.get("stdout_path"),
            sha256=r.get("stdout_sha256"),
            field="passed",
            operator="==",
            expected=True,
            observed=False,
            passed=False,
        )
        for r in receipts
        if not r["passed"]
    ]
    value["field_principles"].update(
        {
            k: "Bind immutable invocation evidence; earlier quarantines and zero independent benefit remain."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = legacy.checksum(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Keep the shipped numerical replay and compare the entire new envelope."""
    try:
        value = json.loads(path.read_bytes())
        if legacy.checksum(value) != value["reproducibility_checksum"]:
            return False
        work = inputs.read(value["primitive_reference"])
        with TemporaryDirectory(prefix="exp8558-replay-", dir="/var/tmp") as directory:
            candidate = Path(directory) / "previous.json"
            atomic_json(
                candidate,
                previous.artifact(
                    work, value["validation_receipts"], Path(value["publication_output"])
                ),
            )
            if not previous.replay(candidate):
                return False
        return bool(
            artifact(work, value["validation_receipts"], Path(value["publication_output"])) == value
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False


def consume(path: Path) -> Json:
    """The actual timing consumer receives only a completely qualified frozen bundle."""
    value = json.loads(path.read_bytes())
    if (
        value["nine_input_native_ready_score"] != 1
        or not value["required_checks_passed"]
        or not replay(path)
    ):
        raise ValueError("nine_input_native_not_ready")
    with TemporaryDirectory(prefix="exp8558-consumer-", dir="/var/tmp") as directory:
        candidate = Path(directory) / "qualified_previous.json"
        work = inputs.read(value["primitive_reference"])
        atomic_json(
            candidate,
            previous.artifact(
                work, value["validation_receipts"], Path(value["publication_output"])
            ),
        )
        return dict(previous.consume(candidate))
