"""REQ-REPORT-8651: admit complete costs without inventing physical performance."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting import hardware_operation_delta_8638 as old
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
ROOT, reference, checked, checksum, failure = (
    old.ROOT,
    old.reference,
    old.checked,
    old.checksum,
    old.failure,
)
NAME, TASK = (
    "experiment_8651_v742_hardware_operation_evidence",
    "exp8651-hardware-operation-evidence",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_evidence_8651.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_evidence_8651.py",
    "python/carnot/reporting/hardware_evidence_runner_8651.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PRODUCERS = [
    (8643, "causal_brier_stream", "learning_cost_ready_score", "delayed_learning"),
    (8647, "installed_complete_cost", "installed_cost_ready_score", "installed_inference"),
    (8645, "complete_source_comparison", "source_comparison_ready_score", "source_extraction"),
]
REQUIRED = {
    "installed_inference": old.OPERATIONS,
    "delayed_learning": [
        *old.OPERATIONS,
        "sparse_coordinate_updates",
        "durable_synchronization",
        "feedback_drain",
    ],
    "source_extraction": [
        "worker_startup_imports",
        "full_text_features",
        "frame_transport",
        "serialization",
        "worker_shutdown",
        "client_shutdown",
    ],
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so the supervisor can see real progress without changing clocks."""
    print(f"[exp8651] phase={phase} completed={completed} pending={pending}", flush=True)


def reduce_spans(value: Json, boundary: str) -> Json:
    """Partition one complete service interval; overlapping or missing work invalidates a bound."""
    spans = value["spans"]
    if (
        value["boundary"] != boundary
        or value["exit_code"] != 0
        or not spans
        or spans[0]["start_ns"] != value["start_ns"]
        or spans[-1]["end_ns"] != value["end_ns"]
    ):
        raise ValueError("complete_service_endpoints")
    seconds = dict.fromkeys(REQUIRED[boundary], 0.0)
    previous = value["start_ns"]
    for span in spans:
        start, end = span["start_ns"], span["end_ns"]
        if (
            type(start) is not int
            or type(end) is not int
            or start != previous
            or end < start
            or span["operation"] not in seconds
        ):
            raise ValueError("disjoint_complete_operation_spans")
        seconds[span["operation"]] += (end - start) / 1e9
        previous = end
    if set(s["operation"] for s in spans) != set(seconds):
        raise ValueError("missing_required_operation")
    total = (value["end_ns"] - value["start_ns"]) / 1e9
    if total <= 0:
        raise ValueError("nonpositive_complete_span")
    compatible = seconds.get("deterministic_spline_109", 0.0)
    remainder = sum(v for k, v in seconds.items() if k != "deterministic_spline_109")
    return dict(
        boundary=boundary,
        cohort=value["cohort"],
        arm=value["arm"],
        total_s=total,
        seconds=seconds,
        compatible_s=compatible,
        unaccelerated_share=remainder / total,
        conditional_ceiling=total / remainder if remainder else None,
        conditional_limit="finite_conditional_bound"
        if remainder
        else "unbounded_conditional_limit",
        achieved_speedup=None,
        required_architecture="A compatible binary64 spline kernel and typed-policy parity; colocate startup, source processing, transport, policy, durable writes and final drain in a persistent service. Measure each changed boundary separately.",
    )


def admit(root: Path, producer: tuple[int, str, str, str]) -> Json:
    """Each branch must authenticate itself; a sibling's missing costs cannot veto it."""
    number, suffix, field, boundary = producer
    path = root / f"results/experiment_{number}_v742_{suffix}.json"
    result: Json = dict(
        experiment_id=number,
        boundary=boundary,
        qualified=False,
        cells=[],
        refs=[],
        checks=[],
        missing_reason="no_qualified_complete_primitive_closure",
    )
    field_name, expected, observed = field, 1, None
    try:
        if path.is_file():
            result["refs"].append(reference(path))
        value = json.loads(path.read_bytes())
        validate_primary(value, path)
        if value["task_id"] != f"exp{number}-{suffix.replace('_', '-')}":
            raise ValueError("producer_task_identity")
        for field_name, expected in [
            (field, 1),
            ("required_checks_passed", True),
            ("admissible_verdict", True),
            ("acceptance_gates", True),
            ("noncritical_findings", True),
        ]:
            observed = {
                "admissible_verdict": value.get("verdict_class")
                in {"positive", "null", "circular_positive"},
                "acceptance_gates": bool(value.get("acceptance_gates"))
                and all(v is True for v in value.get("acceptance_gates", {}).values()),
                "noncritical_findings": all(
                    f["severity"].lower() == "info" for f in value.get("adversarial_findings", [])
                ),
            }.get(field_name, value.get(field_name))
            if type(observed) is not type(expected) or observed != expected:
                raise ValueError(field_name)
        field_name, expected, observed = "bound_publication_receipt", True, False
        side = (
            path.parent
            / "raw"
            / path.stem
            / "validators"
            / (sha256_file(path).split(":")[1] + ".json")
        )
        report = read_bound_sidecar(path, side)
        if (
            report.get("primary_path") != str(path.absolute())
            or report["report"]["passed"] is not True
        ):
            raise ValueError(field_name)
        result["refs"].append(reference(side))
        field_name, expected, observed = (
            "operation_cost_closure",
            "nonempty complete primitive references",
            value.get("operation_cost_closure"),
        )
        if not observed:
            raise ValueError(field_name)
        cells = []
        for ref in observed:
            primitive = checked(ref)
            if not primitive.resolve().is_relative_to((path.parent / "raw" / path.stem).resolve()):
                raise ValueError("foreign_primitive")
            cells.append(
                dict(
                    reduce_spans(json.loads(primitive.read_bytes()), boundary),
                    primitive_reference=reference(primitive),
                )
            )
            result["refs"].append(reference(primitive))
        identities = [(c["cohort"], c["arm"]) for c in cells]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate_cohort_arm")
        result.update(qualified=True, cells=cells, missing_reason=None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["checks"].append(
            dict(
                failure(path, f"exp{number}", field_name, expected, observed),
                passed=False,
                reason=str(error),
            )
        )
    return result


def authenticate(root: Path) -> Json:
    """The pinned board ledger is historical authority, not evidence of current device execution."""
    board = old.authenticate(root)["board"]
    board.pop("authentication_duration_s")
    if board["qualified"]:
        original = board["value"]
        board["refs"].append(reference(Path(original["terminal_validation_sidecar_path"])))
        board["value"] = {
            k: original[k]
            for k in [
                "target_model",
                "board_reopen_requirements",
                "kv260_supported_operations",
                "polarfire_graduation",
                "preserved_quarantines",
                "preserved_historical_outcomes",
            ]
        }
        board["value"]["target_model"] = {
            k: original["target_model"][k]
            for k in [
                "name",
                "parameter_count",
                "checkpoint",
                "checkpoint_head",
                "dtype",
                "feature_order",
            ]
        }
    branches = []
    for producer in PRODUCERS:
        branch = admit(root, producer)
        number, suffix, _, _ = producer
        diagnostic = root / f"results/experiment_{number}_{suffix}.json"
        if diagnostic.is_file():
            branch["refs"].append(reference(diagnostic))
            diagnostic_value = json.loads(diagnostic.read_bytes())
            for gate in diagnostic_value.get("gates_evaluated", []):
                upstream = Path(gate["artifact_path"])
                row = failure(
                    upstream,
                    gate["upstream"],
                    gate["artifact_field"],
                    gate["expected"],
                    gate["actual"],
                )
                row["op"] = gate["op"]
                row["recorded_sha256"] = gate["artifact_sha256"]
                row["receipt_hash_matches"] = row["sha256"] == gate["artifact_sha256"]
                branch["checks"].append(row)
                if upstream.is_file():
                    branch["refs"].append(reference(upstream))
        branches.append(branch)
    return dict(board=board, branches=branches)


def measure(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Freeze exact inputs before validation; no timing or model controller is invoked."""
    progress("authenticate_board_once_no_device_or_model_calls", 0, 1)
    start = time.monotonic_ns()
    evidence = authenticate(root)
    authentication_s = (time.monotonic_ns() - start) / 1e9
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "disk_probe"
    probe.write_bytes(b"disk-backed scratch")
    probe.unlink()
    refs = [*evidence["board"]["refs"], *(r for b in evidence["branches"] for r in b["refs"])]
    snapshots = []
    for index, ref in enumerate(refs):
        destination = raw / "inputs" / f"{index:03d}-{Path(ref['path']).name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked(ref), destination)
        destination.chmod(0o444)
        snapshots.append(dict(reference(destination), original_reference=ref))
    work: Json = dict(
        root=str(root.absolute()),
        run_date=run_date,
        evidence=evidence,
        preconditions_checked=dict(
            python=sys.executable,
            tools={
                p: (ROOT / ".venv/bin" / p).is_file() for p in ["python", "pytest", "ruff", "mypy"]
            },
            memory_available=next(
                s
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            ),
            scratch_path=str(private),
            scratch_writable=True,
            scratch_free_bytes=shutil.disk_usage(private).free,
            optional_gpu_required=False,
            optional_board_required=False,
            board_authentication_s=authentication_s,
            board_authenticated_within_60_seconds=authentication_s <= 60,
        ),
        phase_spans=[
            dict(
                phase="input_authentication_and_snapshot",
                start_ns=start,
                end_ns=time.monotonic_ns(),
                duration_s=(time.monotonic_ns() - start) / 1e9,
            )
        ],
        duration_s=(time.monotonic_ns() - start) / 1e9,
        snapshots=snapshots,
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "ops/exclusion_manifest.yaml",
                "docs/research-notes/v742-hardware-operation-evidence.md",
                "python/carnot/reporting/hardware_operation_delta_8638.py",
                "python/carnot/reporting/primary_publication.py",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("inputs_frozen_measurement_complete", len(refs), 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualify report checks separately from unavailable external costs and physical work."""
    historical_work = deepcopy(work)
    historical_work["evidence"]["board"]["authentication_duration_s"] = work[
        "preconditions_checked"
    ]["board_authentication_s"]
    value = old.build(historical_work, receipts, raw, output)
    board, branches = work["evidence"]["board"], work["evidence"]["branches"]
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(f["severity"].lower() == "info" for f in work.get("adversarial_findings", []))
    )
    admitted = [b for b in branches if b["qualified"]]
    cells = [dict(c, producer=b["experiment_id"]) for b in admitted for c in b["cells"]]
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if len(admitted) != 3 or not board["qualified"]
        else "null"
    )
    value.update(
        experiment_id=8651,
        task_id=TASK,
        milestone="2026.10.742",
        honest_verdict=f"complete_{verdict}_hardware_operation_evidence",
        verdict_class=verdict,
        hardware_mapping_ready_score=int(owned and board["qualified"]),
        intended_count=3,
        completed_count=len(admitted),
        censored_count=3 - len(admitted),
        sample_size_budget=dict(
            unit="separate_qualified_cost_workloads", intended=3, independent_questions=0
        ),
        source_admission_rows=[
            dict(
                experiment_id=b["experiment_id"],
                boundary=b["boundary"],
                qualified=b["qualified"],
                missing_reason=b["missing_reason"],
                checks=b["checks"],
            )
            for b in branches
        ],
        complete_cost_rows=cells,
        missing_cost_rows=[r for b in branches if not b["qualified"] for r in b["checks"]],
        conditional_ceiling_rows=[
            dict(
                boundary=b["boundary"],
                cohort=c.get("cohort"),
                arm=c.get("arm"),
                conditional_ceiling=c.get("conditional_ceiling"),
                conditional_limit=c.get("conditional_limit"),
                total_s=c.get("total_s"),
                compatible_s=c.get("compatible_s"),
                achieved_speedup=None,
                missing_reason=b["missing_reason"],
                interpretation="Conditional removal of compatible spline arithmetic only; all host work remains. This is not achieved speed.",
            )
            for b in branches
            for c in (b["cells"] or [{}])
        ],
        board_status_rows=[
            dict(r, physical_execution_this_invocation=False)
            for r in (board["value"] or {}).get("board_reopen_requirements", [])
        ],
        operation_support_rows=[
            dict(
                boundary=b["boundary"],
                substrate=substrate,
                operation=operation,
                supported_scope=scope,
                native_current_head_support=substrate in {"CPU_sparse", "Rust_arithmetic"}
                and operation == "deterministic_spline_109",
                physical_execution_this_invocation=False,
                cost_admitted=b["qualified"],
            )
            for b in branches
            for substrate, scope in [
                ("CPU_sparse", "109-coefficient spline and sparse updates"),
                (
                    "Rust_arithmetic",
                    "109-coefficient arithmetic; separate34-coefficient kernel is not interchangeable",
                ),
                ("KV260", "quadratic Ising energy k<=5 only"),
                ("PolarFire", "board-local Linux CPU dispatch only; no fabric kernel"),
                (
                    "GateMate",
                    "dated physical change then GM1Ax IDCODE0x20000001 then n16 sample/hash evidence",
                ),
                ("NPU", "unqualified"),
                ("TSU", "unqualified"),
            ]
            for operation in [
                "quadratic_Ising_energy_k_le_5",
                "deterministic_spline_109",
                "distribution_sampling",
                "optimization",
            ]
        ],
        severity_classification_rows=[
            dict(
                f,
                qualification_effect="retained_info"
                if f["severity"].lower() == "info"
                else "disqualified",
            )
            for f in work.get("adversarial_findings", [])
        ],
        hardware_100x_requirements=dict(
            target=100,
            maximum_unaccelerated_share=0.01,
            compatible_dominant_operations_required=True,
            target_met=False,
            observed_unaccelerated_shares=[c["unaccelerated_share"] for c in cells],
            preventing_host_operations=sorted(
                {
                    k
                    for c in cells
                    for k, v in c["seconds"].items()
                    if k != "deterministic_spline_109" and v > 0
                }
            ),
            missing_reason="no_qualified_complete_learning_spans"
            if not branches[0]["qualified"]
            else None,
            architecture_work="Measure a compatible binary64 spline/update kernel; colocate source processing, transport, policy, startup/shutdown and batched durable feedback in a persistent service. Keep changed service cohorts separate.",
        ),
        historical_model_provenance=[
            dict(
                scope="historical_only",
                source=board["refs"][:1],
                field="historical_model_provenance",
                generator_identity="unsloth/Qwen3.8-27B-GGUF",
            )
        ],
        random_seed=7428651,
    )
    for key in [
        "hardware_delta_ready_score",
        "installed_inference_operation_rows",
        "delayed_learning_operation_rows",
        "learning_acceleration_path",
    ]:
        value.pop(key)
    value["new_operation_cost_rows"] = cells
    value["required_checks_passed"] = owned
    value["methodology_note"] = (
        "Read-only authenticated cost mapping. No language model, board, or benchmark runs. Qualified complete primitive closures are reduced separately; unavailable costs remain null."
    )
    value["acceptance_gates"] = dict(
        original_board_authenticated=board["qualified"],
        owned_report_checks=owned,
        complete_new_cost_boundaries=len(admitted) == 3,
        compatible_physical_execution=False,
    )
    value["preserved_quarantines"]["Exp8638"] = (
        "complete_disqualified_no_admitted_hardware_operation_delta; historical verdict and bytes unchanged"
    )
    value["preserved_historical_outcomes"].append(
        "V741 thirteen scheduling slots/seven producers; learning did not run; streaming parity lost original coverage custody; installed prototype qualified but compatibility failed original union after resident_protocol.py changed; Exp8637 sixteen censored redirects and observer action parity without complete horizon or promotion."
    )
    value["numerical_inference_this_invocation"].update(
        historical_numerical_evidence=dict(
            reference=board["refs"][:1],
            fields=[
                "target_model",
                "native_arithmetic_reference",
                "resident_inference_operation_rows",
            ],
            scope="authenticated historical numerical head evidence, not current inference",
        )
    )
    value["branch_gate_values"].pop("hardware_delta_ready_score")
    value["branch_gate_values"]["hardware_mapping_ready_score"] = value[
        "hardware_mapping_ready_score"
    ]
    value["raw_shard_hashes"] += work["snapshots"]
    value["work_reference"] = reference(Path(work.get("work_path", raw / "measurement.json")))
    if value["work_reference"] not in value["raw_shard_hashes"]:
        value["raw_shard_hashes"].append(value["work_reference"])
    value["raw_shard_hashes"] += [
        reference(p)
        for p in sorted((raw / "validation").rglob("*"))
        if p.is_file() and "retained" in p.parts
    ]
    value["field_principles"] = {
        k: "Exact invocation evidence. Missing costs are null. Report qualification, operation compatibility and physical benefit remain separate."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reauthenticate original inputs rather than trusting a self-consistently rehashed summary."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(checked(value["work_reference"]).read_bytes())
        if authenticate(Path(work["root"])) != work["evidence"]:
            return False
        for snapshot in work["snapshots"]:
            if (
                snapshot["sha256"] != snapshot["original_reference"]["sha256"]
                or snapshot["original_reference"] not in value["source_artifact_hashes"]
            ):
                return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            checked(ref)
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
