"""REQ-REPORT-8545: bind repaired execution to original numerical evidence.

The older measurement code already owns the mathematical policy. This adapter
adds complete request evidence and stricter release gates without fitting again.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import nine_input_qualification_8533 as legacy
from carnot.reporting import nine_input_inputs_8533 as inputs
from carnot.reporting import v734_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = legacy.ROOT
NAME = "experiment_8545_v734_nine_input_native_qualification"
TASK = "exp8545-nine-input-native-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_nine_input_8545.py"
OWNED = [
    f"python/carnot/reporting/nine_input_{p}_8545.py"
    for p in ["qualification", "validation", "runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []
__all__ = ["legacy", "inputs"]
CURRENT = "results/experiment_8540_v734_contract_methods.json"
CURRENT_PIN = "sha256:f0fc7fee59e48f83f025e65682c8ec4f5f5f1bd5170e09689746341e61260a1d"
_OPERANDS = inputs.operands


def original_authority() -> list[Json]:
    """Use sealed historical aliases because the live roadmap can change later."""
    protocol = inputs.read(dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN))
    refs = []
    for key in ["v733_design_reference", "v733_activation_reference"]:
        ref = protocol[key]
        inputs.read(ref) if ref["path"].endswith(".json") else inputs.reference(Path(ref["path"]))
        if inputs.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("original_authority_hash")
        refs.append(dict(ref, snapshot_path=ref["path"], exists=True))
    return refs


@contextmanager
def sealed(scratch: Path) -> Iterator[None]:
    """Keep original producer dates and task objects during replay and CLI tests."""
    scratch.mkdir(parents=True, exist_ok=True)
    with frozen_inputs(original_authority(), scratch):
        yield


def requests(head: Json, tape: Json, native: Any) -> list[Json]:
    """Re-extract public text for both arms so residual parity cannot hide a consumer bug."""
    rows = []
    for index, row in enumerate(tape["rows"]):
        reference = json.loads(legacy.numeric.request(head, row))
        for arm, provider in [("python", None), ("native", native)]:
            response = legacy.numeric.request(head, row, provider)
            observed = json.loads(response)
            error = abs(observed["probability"] - reference["probability"])
            rows.append(
                dict(
                    unit_id=row["unit_id"],
                    arm=arm,
                    response=response,
                    probability_error=error,
                    boundary_fallback=observed["boundary_fallback"],
                    passed=error <= 1e-12 * (1 + abs(reference["probability"]))
                    and observed["decision"] == reference["decision"],
                )
            )
        if (index + 1) % 100 == 0:
            legacy.progress("full_text_requests", len(rows), 2 * (len(tape["rows"]) - index - 1))
    return rows


def measure(root: Path, raw: Path) -> Json:
    """Authenticate the prior methods gate before reusing the sealed original workload."""

    def operands(source: Path, destination: Path) -> tuple[Json, Json, list[Json], Json]:
        """Seal only historical reads so current raw evidence keeps its real location."""
        with TemporaryDirectory(prefix="exp8545-inputs-", dir="/var/tmp") as directory:
            scratch = Path(directory)
            with sealed(scratch / "authority"):
                head, tape, refs, upstream = _OPERANDS(source, scratch / "inputs")
        atomic_json(destination / "head.json", head)
        atomic_json(destination / "request_tape.json", tape)
        return dict(head), dict(tape), list(refs), dict(upstream)

    with patch.object(inputs, "operands", operands):
        work = legacy.measure(root, raw)
    toolchain = [
        inputs.reference(Path(p)) for p in work["preconditions"]["tool_paths"].values() if p
    ]
    work["code_refs"].extend(toolchain)
    work["preconditions"]["extension_environment"] = os.environ.get("CARNOT_8533_EXTENSION")
    if work["extension"]:
        work["extension"]["toolchain_hashes"] = toolchain
    work["complete_request_rows"] = []
    try:
        upstream = inputs.read(dict(path=str(root / CURRENT), sha256=CURRENT_PIN))
        terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
        bound = read_bound_sidecar(root / CURRENT, Path(terminal["publication"]["sidecar_path"]))
        if (
            any(
                upstream.get(k) != 1
                for k in ["methods_ready_score", "nine_input_inputs_ready_score"]
            )
            or bound["report"]["passed"] is not True
        ):
            raise ValueError("methods_gate")
        work["source_refs"].append(dict(path=str(root / CURRENT), sha256=CURRENT_PIN))
    except (OSError, ValueError, KeyError) as error:
        for field in ["methods_ready_score", "nine_input_inputs_ready_score"]:
            observed = (
                json.loads((root / CURRENT).read_bytes()).get(field)
                if (root / CURRENT).is_file()
                else None
            )
            work["failures"].append(
                dict(
                    legacy.diagnostic(root / CURRENT, field, observed),
                    check="methods_and_native_inputs",
                    upstream="exp8540-contract-methods",
                    reason=str(error),
                )
            )
    if work["extension"] and not work["failures"]:
        head = inputs.read(work["head_reference"])
        tape = inputs.read(work["request_tape_reference"])
        work["complete_request_rows"] = requests(
            head, tape, legacy.build.load(Path(work["extension"]["path"]))
        )
    work["code_refs"].extend(inputs.reference(ROOT / p) for p in OWNED + [TEST])
    work["original_authority"] = original_authority()
    atomic_json(raw / "complete_requests.json", dict(rows=work["complete_request_rows"]))
    work["complete_requests_reference"] = inputs.reference(raw / "complete_requests.json")
    work["measurement_ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    return dict(work)


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Every required contract must pass before a future timing task can consume bytes."""
    value = legacy.artifact(work, receipts, output)
    required = {
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
    }
    checks = {r["name"]: r["passed"] for r in receipts}
    complete = len(work["complete_request_rows"]) == 1800 and all(
        r["passed"] for r in work["complete_request_rows"]
    )
    ready = (
        value["nine_input_native_ready_score"] == 1
        and complete
        and required <= checks.keys()
        and all(checks[n] for n in required)
        and not work.get("adversarial_findings")
    )
    verdict = (
        "circular_positive"
        if ready
        else "blocked"
        if value["verdict_class"] == "blocked"
        else "disqualified"
    )
    value.update(
        experiment_id=8545,
        task_id=TASK,
        milestone="2026.10.734",
        honest_verdict=f"complete_{verdict}_v734_nine_input_native_qualification",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        nine_input_native_ready_score=int(ready),
        required_checks_passed=ready,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        complete_request_rows=work["complete_request_rows"],
        extension_reference=work["extension"],
        build_receipts=work["extension"].get("receipts", []),
        private_cli_receipts=[r for r in receipts if r["name"] == "private_complete_cli"],
        parity_rows=work["rows"],
        fallback_rows=[r for r in work["rows"] if r.get("boundary_fallback")],
        original_authority_by_suite=work.get("original_authority_by_suite", {}),
        exact_union_receipt=work.get("exact_union_receipt", {}),
        python_coverage=work.get("python_coverage", {}),
        rust_coverage=work.get("rust_coverage", {}),
        preserved_quarantines=[8530, 8533, 8537],
        original_failure_reproduction=work.get("original_failure_reproduction", []),
    )
    value["acceptance_gates"].update(
        complete_request_consumer=complete, all_required_contracts=ready
    )
    value["sample_size_budget"].update(complete_request_arms=1800, timed_requests=0)
    value["duration_s"] = work.get("total_duration_s", value["duration_s"])
    value["native_execution_count"] += sum(
        r["arm"] == "native" for r in work["complete_request_rows"]
    )
    value["boundary_fallback_count"] = sum(
        bool(r.get("boundary_fallback")) for r in work["rows"]
    ) + sum(r["arm"] == "native" and r["boundary_fallback"] for r in work["complete_request_rows"])
    measured = (work["measurement_ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9
    value["phase_spans"] = [
        dict(name="sealed_native_and_complete_text_parity", duration_s=measured),
        dict(
            name="owned_validation_and_prepublication_controls",
            duration_s=max(0, value["duration_s"] - measured),
        ),
    ]
    value["raw_shard_hashes"].append(work["complete_requests_reference"])
    value["field_principles"].update(
        {
            k: "Invocation-bound qualification evidence; no new independent science or speed estimate."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = legacy.checksum(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Recompute original operands as well as the envelope so rehashed changes fail."""
    try:
        value = json.loads(path.read_bytes())
        if legacy.checksum(value) != value["reproducibility_checksum"]:
            return False
        work = inputs.read(value["primitive_reference"])
        with TemporaryDirectory(prefix="exp8545-replay-", dir="/var/tmp") as directory:
            scratch = Path(directory)
            original = legacy.artifact(
                work, value["validation_receipts"], Path(value["publication_output"])
            )
            candidate = scratch / "original.json"
            atomic_json(candidate, original)
            with sealed(scratch / "authority"):
                if not legacy.replay(candidate):
                    return False
            if work["complete_request_rows"]:
                actual = requests(
                    inputs.read(work["head_reference"]),
                    inputs.read(work["request_tape_reference"]),
                    legacy.build.load(Path(work["extension"]["path"])),
                )
                if (
                    actual != work["complete_request_rows"]
                    or inputs.read(work["complete_requests_reference"])["rows"] != actual
                ):
                    return False
        return bool(
            artifact(work, value["validation_receipts"], Path(value["publication_output"])) == value
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False


def consume(path: Path) -> Json:
    """Future timing can use only the exact qualified binary, tape and policy."""
    value = json.loads(path.read_bytes())
    if (
        value["nine_input_native_ready_score"] != 1
        or not value["required_checks_passed"]
        or not replay(path)
    ):
        raise ValueError("nine_input_native_not_ready")
    return dict(
        ready=True,
        extension=value["extension_reference"],
        request_tape=value["request_tape_reference"],
        head=value["head_reference"],
        measurement_kernel=value["measurement_kernel_reference"],
    )
