"""REQ-REPORT-8003: immutable producer evidence keeps board custody independent.

Historical failed operands stay in the output. Optional service absence limits
estimates, while missing required board or sparse bytes closes the current gate.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import fixedpoint_sparse_8003 as f

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAMES = {
    7977: "experiment_7977_v691_hardware_evidence.json",
    7990: "experiment_7990_v692_hardware_evidence.json",
    7996: "experiment_7996_v693_sparse_energy_training.json",
    8002: "experiment_8002_v693_service_cost.json",
}
PINS = {
    7977: dict(
        primary="sha256:15147d4b31b7a1c525eda300e7333942346e05e74efd1f28af28ffe4a5c45d7e",
        terminal="sha256:43993417cc30156e82295a9407593e2bc8a07b70e49d3d69e19defe1de9c1c8d",
    ),
    7990: dict(
        primary="sha256:d3895e97defb556ca0b0b4faba4dea8f5acf31618c73b37c5edc1b359388a7f4",
        terminal="sha256:a54fd0156829136450a33638948a38e4373f70ee240818987de0e955d07e1f66",
    ),
    7996: dict(
        primary="sha256:ab4193d06c43b8abae95d2dc866a544246eff17179ec6fb90f79d7bab8b20bda",
        terminal="sha256:c81b59d9224ce5d5b8c3d0e376a021f8a983af41cebb4a46e93197e20a261748",
        report="sha256:541eb1b400c683079ed30822f827b7b5d8b3a49e60a5e5ec01fd240df727991b",
    ),
    8002: dict(
        primary="sha256:505b9dc0f112bfb7e691a60d77d17397e1d1153130be2094a6a9ed8d161f2d1b",
        terminal="sha256:a0069bfb41fe12ffbb3fa609d8509d3ff3891debccf704012ade3aaf14f43af4",
        report="sha256:160a22d7b0c1977d9d1450930afdeff4c247fc8b1f6ff51d234275b9d5c1f581",
    ),
}


def operand(
    eid: int | str, path: Path, field: str, expected: Any, observed: Any, digest: str | None = None
) -> Json:
    """Every failure names its exact operand instead of replacing missing data by zero."""
    return dict(
        upstream_id=f"exp{eid}",
        artifact_path=str(path),
        artifact_hash=digest,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def bound(root: Path, label: str) -> Path:
    """Private authority copies relocate repository labels while retaining external snapshots."""
    path = Path(label)
    return root / path.relative_to(ROOT) if path.is_relative_to(ROOT) else root / path


def seal(
    root: Path,
    ref: Json,
    directory: Path,
    eid: int,
    checks: list[Json],
    refs: list[Json],
    git_code: bool = False,
) -> Path | None:
    """Hash-matched Git objects provide producer code without trusting mutable files."""
    path = bound(root, ref["path"])
    target = directory / (ref["sha256"].split(":")[-1] + path.suffix)
    directory.mkdir(parents=True, exist_ok=True)
    if git_code:
        relative = Path(ref["path"]).relative_to(ROOT)
        print(f"[exp8003] before_subprocess producer_git_snapshot path={relative}", flush=True)
        child = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, timeout=30
        )
        print(
            f"[exp8003] after_subprocess producer_git_snapshot exit={child.returncode}", flush=True
        )
        content = child.stdout if child.returncode == 0 else None
    else:
        content = path.read_bytes() if path.is_file() else None
    if content is not None:
        target.write_bytes(content)
    digest = sha256_file(target) if content is not None else None
    checks.append(
        operand(
            eid,
            path,
            "producer_snapshot.sha256" if git_code else "sha256",
            ref["sha256"],
            digest,
            digest,
        )
    )
    if digest != ref["sha256"]:
        return None
    refs.append(
        dict(
            path=str(target),
            sha256=digest,
            original_reference=ref,
            producer_id=eid,
            resolution="immutable_git_object" if git_code else "sealed_producer_bytes",
        )
    )
    return target


def primary(root: Path, eid: int, directory: Path, checks: list[Json], refs: list[Json]) -> Json:
    """Pinned primaries bind their terminal reports and eligible producer identity."""
    path = seal(
        root,
        dict(path=str(root / "results" / NAMES[eid]), sha256=PINS[eid]["primary"]),
        directory,
        eid,
        checks,
        refs,
    )
    if path is None:
        return {}
    value: Json = json.loads(path.read_bytes())
    refs[-1].update(
        producer_invocation_date=value.get("execution_date", value.get("run_date")),
        imported_fields=["board_rows"]
        if eid == 7977
        else ["gate_check_summary"]
        if eid == 7990
        else ["checkpoints", "rows"],
    )
    if eid == 7990:
        return value
    field = {
        7977: "hardware_evidence_ready_score",
        7996: "sparse_fit_ready_score",
        8002: "service_measurement_ready_score",
    }[eid]
    for key, expected in (("experiment_id", eid), (field, 1), ("flagged_adversarial", False)):
        if key not in value:
            raise ValueError("upstream_contract:" + key)
        checks.append(operand(eid, path, key, expected, value[key], PINS[eid]["primary"]))
    receipt = value["validation_receipts"]
    passed = (
        receipt.get("required_checks_passed")
        if isinstance(receipt, dict)
        else bool(receipt) and all(r["passed"] for r in receipt if r["required"])
    )
    checks.append(
        operand(
            eid,
            path,
            "validation_receipts.required_checks_passed",
            True,
            passed,
            PINS[eid]["primary"],
        )
    )
    checks.append(
        operand(
            eid,
            path,
            "verdict_class.eligible",
            True,
            value["verdict_class"] in {"positive", "null", "circular_positive"},
            PINS[eid]["primary"],
        )
    )
    terminal = seal(
        root,
        dict(path=value["terminal_validation_sidecar_path"], sha256=PINS[eid]["terminal"]),
        directory,
        eid,
        checks,
        refs,
    )
    if terminal is None:
        return value
    report = json.loads(terminal.read_bytes())
    checks.append(
        operand(
            eid,
            terminal,
            "primary_sha256",
            PINS[eid]["primary"],
            report.get("primary_sha256", report.get("candidate_sha256")),
            PINS[eid]["terminal"],
        )
    )
    if "sidecar_path" in report:
        sidecar = seal(
            root,
            dict(path=report["sidecar_path"], sha256=PINS[eid]["report"]),
            directory,
            eid,
            checks,
            refs,
        )
        report = json.loads(sidecar.read_bytes())["report"] if sidecar else {}
    checks.append(
        operand(eid, terminal, "terminal.passed", True, report.get("passed"), PINS[eid]["terminal"])
    )
    for r in report.get("receipts", report.get("reports", [])):
        seal(root, dict(path=r["log_path"], sha256=r["log_sha256"]), directory, eid, checks, refs)
    return value


def authenticate(root: Path, directory: Path) -> Json:
    """Required custody and sparse branches qualify separately from optional service."""
    checks: list[Json] = []
    refs: list[Json] = []
    prior = primary(root, 7977, directory, checks, refs)
    boards = deepcopy(prior.get("board_rows", fixture()["boards"]))
    historical = deepcopy(prior.get("historical_required_failures", []))
    historical += deepcopy(prior.get("historical_failures", []))
    historical += deepcopy(prior.get("historical_7901_affected_failures", []))
    for r in prior.get("preconditions_checked", []):
        if r["artifact_field"] == "sha256":
            seal(
                root,
                dict(path=r["artifact_path"], sha256=r["expected"]),
                directory,
                7977,
                checks,
                refs,
            )
    for board in boards:
        if "source_hash" in board:
            seal(
                root,
                dict(path=board["source_path"], sha256=board["source_hash"]),
                directory,
                7977,
                checks,
                refs,
            )
    custody = bool(prior) and all(c["passed"] for c in checks)
    for board in boards:
        board.update(
            custody_valid=custody, current_hardware_execution=False, compatible_sparse_kernel=False
        )
    sparse = primary(root, 7996, directory, checks, refs)
    assets = {}
    for key in ("heads", "inputs"):
        if sparse:
            path = seal(root, sparse["checkpoints"][key], directory, 7996, checks, refs)
            if path:
                assets[key] = json.loads(path.read_bytes())
    for ref in sparse.get("code_config_hashes", []):
        seal(
            root,
            ref,
            directory,
            7996,
            checks,
            refs,
            git_code=Path(ref["path"]).is_relative_to(ROOT) and Path(ref["path"]).suffix == ".py",
        )
    sparse_ready = bool(assets) and all(
        c["passed"] for c in checks if c["upstream_id"] == "exp7996"
    )
    old_checks: list[Json] = []
    old = primary(root, 7990, directory, old_checks, refs)
    historical += old.get("gate_check_summary", []) + [c for c in old_checks if not c["passed"]]
    optional: list[Json] = []
    service = primary(root, 8002, directory, optional, refs)
    snapshots = {
        r.get("original_reference", {}).get("path"): r
        for r in service.get("measurement_code_snapshot", [])
    }
    for ref in (
        service.get("code_config_hashes", [])
        + service.get("raw_shard_hashes", [])
        + service.get("source_artifact_hashes", [])
    ):
        snapshot = snapshots.get(ref["path"], ref)
        optional.append(
            operand(
                8002,
                Path(snapshot["path"]),
                "producer_snapshot.declared_hash",
                ref["sha256"],
                snapshot["sha256"],
                snapshot["sha256"],
            )
        )
        seal(root, snapshot, directory, 8002, optional, refs)
    service_ready = bool(service) and all(c["passed"] for c in optional)
    return dict(
        boards=boards,
        board_custody_ready_score=int(custody),
        sparse_available=sparse_ready,
        head=assets.get("heads", {}).get("heads", {}).get("spline", [None])[0]
        if sparse_ready
        else None,
        data=assets.get("inputs", {}).get("data") if sparse_ready else None,
        service_available=service_ready,
        service_rows=service.get("rows", []) if service_ready else [],
        checks=checks,
        optional_service_blockers=[c for c in optional if not c["passed"]],
        optional_service_checks=optional,
        historical_failed_operands=historical,
        cited_upstream_artifacts=refs,
        fixture=False,
    )


def fixture() -> Json:
    """A separable arithmetic control supplies headroom without natural-data claims."""
    head = dict(
        arm="spline",
        seed=17,
        parameters=[0.01] * 108 + [0.1],
        decay_scale=1.0,
        temperature=2.0,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
    )
    data: Json = dict(fit=[], tune=[])
    for index in range(12):
        data["fit" if index < 8 else "tune"].append(
            dict(
                family_id=f"fixture-{index}",
                source_cluster_id=f"source-{index}",
                q=0.1 + index * 0.07,
                features=[0.15 + index * 0.05] * 8,
                y=int(index > 5),
            )
        )
    boards = [
        dict(
            board=name,
            scope=scope,
            processor_class=processor,
            k_max=5 if name == "KV260" else None,
            blocker="0xffffffff" if name == "GateMate" else None,
            custody_valid=True,
            receipt_date="20260913",
            current_hardware_execution=False,
            compatible_sparse_kernel=False,
        )
        for name, scope, processor in (
            ("KV260", "SSH authenticated quadratic Ising fabric, k_max<=5", "fpga_fabric"),
            ("PolarFire", "Linux CPU dispatch only", "linux_cpu"),
            ("GateMate", "physical/JTAG blocker; operator hardware change required", "unavailable"),
        )
    ]
    return dict(
        boards=boards,
        board_custody_ready_score=1,
        sparse_available=True,
        head=head,
        data=data,
        service_available=True,
        service_rows=[
            dict(
                arm="spline",
                case="sparse_update_durable",
                cached_total_s=1.0,
                complete_total_s=2.0,
                exclusive_phase_spans=dict(
                    head_prediction=100000000,
                    feedback_processing=50000000,
                    typed_decision=10000000,
                    storage_fsync=10000000,
                ),
            )
        ],
        checks=[],
        optional_service_blockers=[],
        historical_failed_operands=[],
        cited_upstream_artifacts=[],
        fixture=True,
    )


def reduce(plan: Json) -> Json:
    """CPU compatibility and analytical placement cannot assert a measured device gain."""
    boards = deepcopy(plan["boards"])
    ready = bool(plan["board_custody_ready_score"] and plan["sparse_available"])
    numeric = (
        f.evaluate(plan["head"], plan["data"])
        if plan["sparse_available"]
        else dict(
            fixed_point_rows=[],
            quantization_ready_score=0,
            quantization_gates={},
            positive_control_results=dict(working=False, scope="not_run_missing_sparse_evidence"),
            independent_source_groups=0,
        )
    )
    service_rows = [r for r in plan["service_rows"] if r["arm"] == "spline"]
    total = sum(r["complete_total_s"] for r in service_rows) if plan["service_available"] else None
    cached = sum(r["cached_total_s"] for r in service_rows) if plan["service_available"] else None
    component = sum(
        (
            r["exclusive_phase_spans"].get("head_prediction", 0)
            + r["exclusive_phase_spans"].get("feedback_processing", 0)
        )
        / 1e9
        for r in service_rows
    )
    fraction = component / total if total else None
    placements = [
        dict(
            board=board["board"],
            operation=op,
            known_scope=board["scope"],
            authenticated=board["custody_valid"],
            implemented_sparse_kernel=False,
            compatible_fraction=0,
            hardware_execution_measured=False,
            reason="Quadratic Ising overlay does not implement arbitrary spline lookup or updates"
            if board["board"] == "KV260"
            else "Linux CPU dispatch only"
            if board["board"] == "PolarFire"
            else "Physical/JTAG 0xffffffff requires operator change",
        )
        for board in boards
        for op in ("sparse_basis_lookup", "additions", "multiplications", "durable_writes")
    ]
    fixed_rows = numeric["fixed_point_rows"]
    rows = deepcopy(fixed_rows)
    for board in boards:
        rows.append(
            dict(
                family_id=board["board"],
                arm="board_custody_audit",
                seed=None,
                numerator=int(board["custody_valid"]),
                denominator=1,
                eligibility=True,
                failure_status=not board["custody_valid"],
                censor_status=False,
                independent=0,
                board=board["board"],
                scope=board["scope"],
            )
        )
    counts = dict(
        intended=len(rows),
        eligible=len(rows),
        started=len(rows),
        completed=len(rows),
        excluded=0,
        failed=sum(bool(r["failure_status"]) for r in rows),
        censored=0,
        independent=numeric["independent_source_groups"],
        independent_definition="Distinct fit/tune source groups; formats and fixtures add no independent sources",
    )
    return dict(
        honest_verdict="complete_circular_positive_cpu_fixture"
        if ready and plan["fixture"]
        else "complete_null_cpu_sparse_boundary_no_device_execution"
        if ready
        else "complete_blocked_required_board_or_sparse_custody",
        verdict_class="circular_positive"
        if ready and plan["fixture"]
        else "null"
        if ready
        else "blocked",
        hardware_evidence_ready_score=int(ready),
        board_custody_ready_score=plan["board_custody_ready_score"],
        board_rows=boards,
        workload_placement_rows=placements,
        modeled_100x_bound=1.0 if total else None,
        ideal_amdahl_bound=1.0 if total else None,
        counterfactual_future_sparse_kernel_bounds=dict(
            compatible_fraction=fraction,
            complete_service_s=total,
            cached_service_s=cached,
            sparse_component_s=component if total else None,
            ideal_amdahl_bound=1 / (1 - fraction) if fraction is not None else None,
            modeled_100x_bound=1 / (1 - fraction + fraction / 100)
            if fraction is not None
            else None,
            scope="unimplemented_future_kernel_optimistic_estimate",
        ),
        transfer_assumptions=dict(
            transfer_s=None,
            assumed_transfer_s=0,
            readout="host typed decision retained",
            durable_writes="host serialization and file/directory fsync retained",
            host_scaler="float64 retained",
            assumption="optimistic zero transfer; no measured hardware speedup",
            authenticated_board_compatible_fraction=0,
        ),
        optional_service_blockers=plan["optional_service_blockers"],
        historical_failed_operands=plan["historical_failed_operands"],
        gate_check_summary=[c for c in plan["checks"] if not c["passed"]],
        preconditions_checked=plan["checks"],
        cited_upstream_artifacts=plan["cited_upstream_artifacts"],
        rows=rows,
        sample_size_budget=counts,
        trained_head_specs=[
            dict(
                arm="spline", parameter_count=109, seed=17, pretrained=False, current_fitting=False
            )
        ]
        if plan["sparse_available"]
        else [],
        acceptance_gate_results=dict(
            validity=ready,
            benefit=False,
            device_execution=False,
            quantization=numeric["quantization_gates"],
        ),
        unqualified_substrates=[
            dict(substrate=s, qualified=False, evidence=None)
            for s in ("NPU", "Extropic TSU", "larger FPGA")
        ],
        integration_scheduled=False,
        hardware_speedup_claimed=False,
        current_device_execution_count=0,
        **numeric,
    )


def replay(value: Json) -> Json:
    """Recompute primitive rows and verify receipt bytes instead of trusting summaries."""
    input_ref = value.get("replay_input_reference")
    if input_ref:
        path = Path(input_ref["path"])
        if (
            sha256_file(path) != input_ref["sha256"]
            or json.loads(path.read_bytes()) != value["replay_inputs"]
        ):
            raise ValueError("input_drift")
    for ref in (
        value["replay_inputs"]["cited_upstream_artifacts"]
        + value.get("raw_shard_hashes", [])
        + value.get("code_config_hashes", [])
    ):
        path = Path(ref["path"])
        if not path.is_file() or sha256_file(path) != ref["sha256"]:
            raise ValueError("receipt_drift:" + str(path))
    for ref in value.get("validation_receipts", []):
        if ref.get("log_path") and sha256_file(Path(ref["log_path"])) != ref["log_sha256"]:
            raise ValueError("receipt_drift:validation")
    fresh = reduce(value["replay_inputs"])
    disqualified = value["verdict_class"] == "disqualified"
    for key, expected in fresh.items():
        if disqualified and key in {
            "honest_verdict",
            "verdict_class",
            "hardware_evidence_ready_score",
            "gate_check_summary",
        }:
            continue
        if value.get(key) != expected:
            raise ValueError("reduction_drift:" + key)
    if disqualified and (
        value["hardware_evidence_ready_score"] != 0
        or not any(r.get("required") and not r["passed"] for r in value["validation_receipts"])
    ):
        raise ValueError("reduction_drift:owned_failure")
    return dict(
        passed=True, rows_checksum=canonical_hash(fresh["rows"]), row_count=len(fresh["rows"])
    )
