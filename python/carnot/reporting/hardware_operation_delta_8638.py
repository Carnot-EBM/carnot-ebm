"""REQ-REPORT-8638: a missing cost producer cannot establish hardware benefit."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8638_v741_hardware_operation_delta", "exp8638-hardware-operation-delta"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_delta_8638.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_delta_8638.py",
    "python/carnot/reporting/hardware_delta_runner_8638.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
BOARD = "results/experiment_8625_v740_hardware_operation_boundary.json"
BOARD_PIN = "sha256:2bc960c8b303ebc9b6bc2f7e8c81b7e9b115b1657ef10f50e47b11d52a4e80f9"
PRODUCERS = [
    (8630, "causal_brier_stream", "learning_cost_ready_score"),
    (8635, "installed_complete_cost", "installed_cost_ready_score"),
]
OPERATIONS = [
    "client_lifecycle",
    "worker_startup_imports",
    "full_text_features",
    "frame_transport",
    "scaling",
    "offset",
    "deterministic_spline_109",
    "normalization_fallback",
    "typed_policy",
    "serialization",
    "worker_shutdown",
    "client_shutdown",
]
BOARDS = ["CPU_sparse", "Rust_arithmetic", "KV260", "PolarFire", "GateMate", "NPU", "TSU"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervisor can distinguish progress from a stalled child."""
    print(f"[exp8638] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind exact bytes instead of relying on a mutable file name."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def checked(ref: Json) -> Path:
    """Missing and changed operands must stop replay before claims are reconstructed."""
    path = Path(ref["path"])
    if not path.is_file() or sha256_file(path) != ref["sha256"]:
        raise ValueError("operand_hash:" + str(path))
    return path


def checksum(value: Json) -> str:
    """Exclude the checksum itself so the same primitive reduction has one digest."""
    return canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})


def failure(path: Path, upstream: str, field: str, expected: Any, observed: Any) -> Json:
    """A blocker names the precise absent operand or failed comparison."""
    return dict(
        check=field,
        upstream=upstream,
        path=str(path.absolute()),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def authenticate(root: Path) -> Json:
    """Read only the board ledger and the two branch inputs; never invoke their controllers."""
    start = time.monotonic()
    path = root / BOARD
    board: Json = dict(qualified=False, value=None, refs=[], checks=[])
    try:
        checked(dict(path=str(path), sha256=BOARD_PIN))
        value = json.loads(path.read_bytes())
        validate_primary(value, path)
        envelope = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        side = Path(envelope["publication"]["sidecar_path"])
        if (
            value["required_checks_passed"] is not True
            or read_bound_sidecar(path, side)["report"]["passed"] is not True
        ):
            raise ValueError("board_publication")
        checkpoint = checked(value["target_model"]["checkpoint"])
        board.update(
            qualified=True,
            value=value,
            refs=[reference(path), reference(side), reference(checkpoint)],
        )
    except (OSError, ValueError, KeyError, TypeError):
        board["checks"].append(
            failure(
                path,
                "exp8625-hardware-operation-boundary",
                "original_board_ledger_sha256",
                BOARD_PIN,
                sha256_file(path) if path.is_file() else None,
            )
        )
    board["authentication_duration_s"] = time.monotonic() - start
    branches = []
    for number, suffix, field in PRODUCERS:
        path = root / f"results/experiment_{number}_v741_{suffix}.json"
        diagnostic = root / f"results/experiment_{number}_{suffix}.json"
        refs, checks = [], [failure(path, f"exp{number}", field, 1, None)]
        if path.is_file():
            refs.append(reference(path))
            checks.append(
                failure(
                    path, f"exp{number}", "registered_qualified_complete_primitives", True, False
                )
            )
        if diagnostic.is_file():
            refs.append(reference(diagnostic))
            for gate in json.loads(diagnostic.read_bytes()).get("gates_evaluated", []):
                checks.append(
                    failure(
                        diagnostic,
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["expected"],
                        gate["actual"],
                    )
                )
        branches.append(
            dict(
                experiment_id=number,
                qualified=False,
                refs=refs,
                checks=checks,
                missing_reason="no_admitted_new_complete_cost_boundary",
            )
        )
    return dict(board=board, branches=branches)


def measure(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Authenticate within the first phase; disk checks do not require optional devices."""
    progress("authenticate_board_once", 0, 1)
    start = time.monotonic_ns()
    evidence = authenticate(root)
    progress("board_authenticated_no_device_commands", 1, 0)
    private.mkdir(parents=True, exist_ok=True)
    probe = private / "disk_probe"
    probe.write_bytes(b"disk-backed scratch")
    probe.unlink()
    memory = next(
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    preconditions = dict(
        python=sys.executable,
        tools={
            name: (ROOT / ".venv/bin" / name).is_file()
            for name in ["python", "pytest", "ruff", "mypy"]
        },
        memory_available=memory,
        scratch_path=str(private),
        scratch_writable=True,
        scratch_free_bytes=shutil.disk_usage(private).free,
        optional_gpu_required=False,
        optional_board_required=False,
    )
    elapsed = (time.monotonic_ns() - start) / 1e9
    work = dict(
        root=str(root.absolute()),
        run_date=run_date,
        evidence=evidence,
        preconditions_checked=preconditions,
        duration_s=elapsed,
        phase_spans=[
            dict(
                phase="preconditions_and_input_authentication",
                start_ns=start,
                end_ns=time.monotonic_ns(),
                duration_s=elapsed,
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "ops/exclusion_manifest.yaml",
                "docs/research-notes/v741-hardware-operation-delta.md",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", 2, 0)
    return work


def matrices() -> tuple[list[Json], list[Json]]:
    """Keep each complete boundary visible even though none has qualified clocks."""
    result = []
    for boundary, extra in [
        ("installed_inference", []),
        (
            "delayed_learning",
            ["sparse_coordinate_updates", "durable_synchronization", "feedback_drain"],
        ),
    ]:
        result.append(
            [
                dict(
                    boundary=boundary,
                    operation=op,
                    substrate=board,
                    host_clock_s=None,
                    device_clock_s=None,
                    compatible_physical_execution=False,
                    cost_admitted=False,
                    missing_reason="no_admitted_new_complete_cost_boundary",
                    compatibility="CPU/Rust implement sparse updates and deterministic spline arithmetic; KV260 k<=5 quadratic Ising cannot evaluate this spline or durable host work; PolarFire dispatch is Linux CPU only; other devices lack supported execution",
                )
                for board in BOARDS
                for op in [*OPERATIONS, *extra]
            ]
        )
    return result[0], result[1]


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A blocked terminal record preserves missing costs without repeating the old study."""
    board = work["evidence"]["board"]
    original = board["value"] or {}
    checks = [*board["checks"], *(r for b in work["evidence"]["branches"] for r in b["checks"])]
    checks.append(
        failure(
            Path(work["root"]) / BOARD,
            "exp8625",
            "board_authentication_within_60_seconds",
            True,
            board["authentication_duration_s"] <= 60,
        )
    )
    for receipt in receipts:
        if receipt.get("unchanged_external_info"):
            checks.append(
                failure(
                    Path(receipt["stdout_path"]),
                    "shared_adversarial_verify",
                    "hardware_delta_readiness_name_info",
                    None,
                    "IMPLAUSIBLE_PERFECT: numeric readiness gate is not a measured delta",
                )
            )
    passed = bool(receipts) and all(
        r["passed"] or r.get("unchanged_external_info", False) for r in receipts
    )
    verdict = "blocked" if passed else "disqualified"
    resident, learning = matrices()
    refs = [*board["refs"], *(r for b in work["evidence"]["branches"] for r in b["refs"])]
    logs = [
        dict(path=r[k], sha256=r[k.replace("_path", "_sha256")])
        for r in receipts
        for k in ["stdout_path", "stderr_path"]
        if k in r
    ]
    retained = [
        reference(p)
        for p in sorted((raw / "validation").glob("*"))
        if p.is_file()
        and (
            p.name.startswith(".coverage")
            or p.name in {"original_nodes.json", "coverage.json", "receipts.json"}
        )
    ]
    value: Json = dict(
        experiment_id=8638,
        task_id=TASK,
        milestone="2026.10.741",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_no_admitted_hardware_operation_delta",
        verdict_class=verdict,
        gate_check_summary=checks,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            current_llm_invocations=0,
            model_loads_attempted=0,
            model_loads_completed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
        ),
        historical_model_provenance=[
            dict(
                scope="historical_only",
                source=board["refs"][:1],
                identities=original.get("historical_model_provenance", []),
            )
        ],
        target_model={
            k: original.get("target_model", {}).get(k)
            for k in [
                "name",
                "parameter_count",
                "checkpoint",
                "checkpoint_head",
                "dtype",
                "feature_order",
            ]
        }
        if board["qualified"]
        else None,
        numerical_inference_this_invocation=dict(
            small_head_evaluations=0,
            sparse_updates=0,
            checkpoint_authenticated=board["qualified"],
            explanation="Current work authenticates the small head only; historical deterministic spline numerical inference is arithmetic, distinct from language-model calls.",
        ),
        rows=[
            dict(
                unit_id=f"exp{b['experiment_id']}",
                arm="new_complete_cost",
                metric="complete_operation_cost_s",
                value=None,
                status="censored",
                missing_reason=b["missing_reason"],
            )
            for b in work["evidence"]["branches"]
        ],
        intended_count=2,
        completed_count=0,
        failed_count=0,
        censored_count=2,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            unit="qualified_new_cost_boundaries", intended=2, independent_questions=0
        ),
        verifier_is_oracle=False,
        exposure_scope="read_only_original_evidence_no_new_questions",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=bool(work.get("adversarial_findings", [])),
        adversarial_findings=work.get("adversarial_findings", []),
        acceptance_gates=dict(
            original_board_authenticated=board["qualified"],
            new_cost_boundary_admitted=False,
            disjoint_new_operation_spans=False,
            compatible_physical_execution=False,
            owned_checks=passed,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7418638,
        source_artifact_hashes=refs,
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[reference(raw / "measurement.json"), *logs, *retained],
        cited_upstream_artifacts=refs,
        publication_output=str(output),
        work_reference=reference(raw / "measurement.json"),
        hardware_delta_ready_score=0,
        physical_hardware_ready_score=0,
        NPU_qualified=0,
        TSU_qualified=0,
        nfr01_score=0,
        new_operation_cost_rows=[],
        conditional_ceiling_rows=[
            dict(
                boundary=b,
                zero_cost_ceiling=None,
                missing_reason="no_new_qualified_disjoint_complete_spans",
                required_proof="nonoverlapping full lifecycle spans; remove compatible arithmetic exactly once and retain all fixed host/transfer work",
            )
            for b in ["delayed_learning", "installed_inference"]
        ],
        installed_inference_operation_rows=resident,
        delayed_learning_operation_rows=learning,
        learning_acceleration_path=dict(
            current="CPU/Rust sparse coordinate updates for the 109-coefficient nine-input spline; at most37 local data-gradient touches and one lazy decay write. Durable synchronization and feedback drain remain CPU work. The separate34-coefficient Rust kernel is not this head.",
            measured_dominant_work=None,
            missing_reason="Exp8630 qualified complete learning clocks missing",
            historical_inference_dominant_work=original.get("acquisition_decision", {}).get(
                "measured_dominant_operations"
            ),
            future_100x_prerequisite="Operation-compatible implementation of measured dominant work, binary64 and typed-policy parity, complete disjoint host/transfer/sync clocks. Fixed host fraction must be below0.01 for any finite100x gain.",
            target_met=False,
            fixed_host_cost_prevents_100x=None,
            changed_deployment_boundary="If fixed host cost is >=1%, study a persistent colocated learning service with batched feedback and durable writes; measure that changed boundary separately.",
        ),
        acquisition_prerequisites=dict(
            purchase_authorized=False,
            recommended=False,
            required=[
                "compatible dominant operation",
                "authenticated physical access and parity",
                "complete host transfer update and synchronization cost",
            ],
        ),
        board_reopen_requirements=original.get("board_reopen_requirements", []),
        kv260_supported_operations=original.get("kv260_supported_operations", []),
        polarfire_graduation=original.get("polarfire_graduation"),
        prior_board_ledger_reference=dict(
            path=str(Path(work["root"]) / BOARD),
            sha256=BOARD_PIN,
            reused_unchanged=board["qualified"],
        ),
        device_commands_executed=0,
        literature_operation_mapping=[
            *original.get("literature_operation_mapping", []),
            dict(
                source="research-references.md",
                operation="online_KAN_sparse_local_spline_updates",
                limit="Local support motivates sparse CPU/Rust work. Paper/device ratios are not measured Carnot performance; deterministic evaluation, distribution sampling and argmin remain different operations.",
                extropic_exclusions=["final_dense_logits", "inter_device_transfer"],
                local_speedup=None,
            ),
        ],
        preserved_quarantines=original.get("preserved_quarantines", {}),
        preserved_historical_outcomes=[
            *original.get("preserved_historical_outcomes", []),
            "V740 twelve slots/nine producers; Exp8616 original child shards missing, mapping94=92 not execution; learning blocked; Exp8620 190 responses/128 exposed groups,55 exceed128 windows; Exp8622 lifecycle CI95[2.6021614,2.7140500], native/Python CI95[1.1799076,1.2584954], NFR01 unmet; Exp8624 observer qualification not supervisor benefit",
        ],
        methodology_note="No new model, board or performance measurement. New-cost producers did not qualify. Prior board facts are reused by reference; all cost-dependent conclusions remain unmeasured.",
        newly_owned_coverage=work.get("owned_coverage"),
        exact_union_receipt=work.get("exact_union_receipt"),
    )
    value["branch_gate_values"] = {
        key: value[key]
        for key in [
            "hardware_delta_ready_score",
            "physical_hardware_ready_score",
            "NPU_qualified",
            "TSU_qualified",
            "independent_generalization_score",
            "generalized_learning_benefit_score",
            "nfr01_score",
        ]
    }
    value["field_principles"] = {
        key: "Invocation-bound original evidence; missing costs are null. Qualification does not grant scientific or physical benefit."
        for key in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Independent original pins reject both edited summaries and rehashed input claims."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(checked(value["work_reference"]).read_bytes())
        evidence = authenticate(Path(work["root"]))
        for item in [evidence, work["evidence"]]:
            item["board"].pop("authentication_duration_s")
        if evidence != work["evidence"]:
            return False
        work = json.loads(checked(value["work_reference"]).read_bytes())
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
