"""REQ-REPORT-8693: bind integer feasibility to costs without inventing device speed."""

from __future__ import annotations

from datetime import datetime, timedelta, UTC
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8679 as old
from carnot.reporting import hardware_boundary_costs_8679 as costs
from carnot.reporting import spline_precision_study_8692 as precision
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import reader_receipt, validate_primary

Json = dict[str, Any]
ROOT, reference, checked, checksum = old.ROOT, old.reference, old.checked, old.checksum
NAME = "experiment_8693_v745_hardware_operation_boundary"
TASK = "exp8693-hardware-operation-boundary"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_boundary_8693.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_boundary_8693.py",
    "python/carnot/reporting/hardware_boundary_runner_8693.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PRODUCERS = [
    (8685, "v745", "causal_brier_stream", "learning_cost_ready_score", "delayed_learning"),
    (
        8692,
        "v745",
        "spline_precision_feasibility",
        "precision_mapping_ready_score",
        "integer_precision",
    ),
    (8689, "v745", "installed_batch_cost", "batch_cost_ready_score", "installed_inference"),
]
BASE_BUILD = old.build
BOARD_PIN = "sha256:66c07c928fcec102fceff878087e1fbc6330ec2d1f8c34c6bde3712ed0d27f07"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the parent can distinguish blocked work from silence."""
    print(f"[exp8693] phase={phase} completed={completed} pending={pending}", flush=True)


def producer_authentication(value: Json, path: Path) -> list[Json]:
    """Original producer construction and terminal bytes outrank a consumer's coverage.

    Precision was already numerically replayed by its producer. Rebuilding that
    report authenticates its operands without rerunning the numerical benchmark.
    """
    refs = [*value["code_config_hashes"], *value["source_artifact_hashes"]]
    for ref in refs:
        checked(ref)
    if value["experiment_id"] == 8692:
        work = json.loads(checked(value["primitive_reference"]).read_bytes())
        receipts = json.loads(checked(value["validation_reference"]).read_bytes())["receipts"]
        anchor = path.parent / "raw" / path.stem
        seals = [
            reference(anchor / folder / (value["invocation_id"] + ".json"))
            for folder in ["seals", "validation_seals"]
        ]
        measurement, validation = [json.loads(checked(ref).read_bytes()) for ref in seals]
        if (
            measurement["measurement"] != value["primitive_reference"]
            or measurement["code_refs"] != work["code_refs"]
            or validation["validation"] != value["validation_reference"]
        ):
            raise ValueError("original_precision_independent_seals")
        refs += seals
        if precision.build(work, receipts, Path(value["publication_output"])) != value:
            raise ValueError("original_precision_producer_construction")
        inputs = precision.authenticate()
        if inputs["failures"]:
            raise ValueError("numerical_checkpoint_authentication")
        refs += [value["primitive_reference"], value["validation_reference"], *inputs["refs"]]
    else:
        cli = ROOT / "scripts/experiments" / (path.stem + ".py")
        if not value["code_config_hashes"] or reference(cli) not in value["code_config_hashes"]:
            raise ValueError("missing_declared_producer_source")
        progress("before_original_producer_cold_replay")
        result = subprocess.run(
            [sys.executable, "-u", str(cli), "--cold-replay", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        progress("after_original_producer_cold_replay")
        if result.returncode != 0:
            raise ValueError("original_cost_producer_replay")
    return refs


def admit(root: Path, producer: tuple[int, str, str, str, str]) -> Json:
    """A missing cost branch cannot veto an independently authenticated precision row."""
    number, version, suffix, field, boundary = producer
    path = root / f"results/experiment_{number}_{version}_{suffix}.json"
    if number != 8692:
        branch = old.admit(root, producer)
    else:
        branch = dict(
            experiment_id=number,
            boundary=boundary,
            qualified=False,
            cells=[],
            refs=[],
            checks=[],
            acceptance_shape=None,
            missing_reason=None,
        )
    state, observed, expected = "missing_file", None, 1
    try:
        value = json.loads(path.read_bytes())
        if number == 8692:
            branch["refs"].append(reference(path))
            validate_primary(value, path)
            observed = value.get(field)
            state = (
                "missing_field"
                if field not in value
                else "observed_zero"
                if type(observed) is int and observed == 0
                else "observed"
            )
            if type(observed) is not int or observed != 1:
                raise ValueError(field)
            field, expected, observed = "qualified_producer_checks", True, False
            shape = costs.acceptance(value["acceptance_gates"])
            branch["acceptance_shape"] = shape
            if (
                value["required_checks_passed"] is not True
                or value["flagged_adversarial"] is not False
                or value["task_id"] != precision.TASK
                or value["verdict_class"] not in {"null", "positive", "circular_positive"}
                or not shape["qualified"]
                or any(f["severity"] != "info" for f in costs.flags(value["adversarial_findings"]))
            ):
                raise ValueError(field)
            state, field = "changed_bytes", "authenticated_terminal_producer"
            branch["refs"].append(old.old.receipt(path))
            terminal = reader_receipt(value["task_id"], path.parent, field=producer[3])
            if not terminal["passed"] or terminal["gate_path"] != str(path.absolute()):
                raise ValueError("terminal_reader_identity")
            branch.update(qualified=True, terminal_reader=terminal)
        if branch["qualified"]:
            state, field = "changed_bytes", "original_producer_authentication"
            branch["refs"] += producer_authentication(value, path)
            if number == 8692:
                branch.update(
                    integer_precision_rows=value["rows"],
                    operation_inventory=value["operation_inventory"],
                    accumulator_bounds=value["accumulator_bounds"],
                    checkpoint_authenticated=True,
                )
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        branch.update(qualified=False, cells=[], missing_reason=str(error))
        if not branch["checks"]:
            branch["checks"].append(
                dict(
                    old.old.previous.failure(path, f"exp{number}", field, expected, observed),
                    passed=False,
                    observed_state=state,
                    reason=str(error),
                )
            )
    return branch


def authenticate(root: Path) -> Json:
    """Reuse pinned board authority; a blocked historical map is no scientific gate."""
    board = old.ORIGINAL_AUTHENTICATE(root)["board"]
    pins = {
        **old.HISTORICAL_INPUTS,
        "results/experiment_8659_v743_complete_source_comparison.json": old.SOURCE_PIN,
        "results/experiment_8679_v744_hardware_operation_boundary.json": BOARD_PIN,
    }
    for relative, digest in pins.items():
        path = root / relative
        try:
            checked(dict(path=str(path), sha256=digest))
            board["refs"] += [reference(path), old.old.receipt(path)]
        except (OSError, ValueError, KeyError, TypeError) as error:
            board["qualified"] = False
            board["checks"].append(
                dict(
                    old.old.previous.failure(
                        path,
                        path.stem,
                        "original_sha256",
                        digest,
                        reference(path)["sha256"] if path.is_file() else None,
                    ),
                    passed=False,
                    observed_state="changed_bytes" if path.is_file() else "missing_file",
                    reason=str(error),
                )
            )
    return dict(board=board, branches=[admit(root, p) for p in PRODUCERS])


def measure(root: Path, raw: Path, private: Path, run_date: str) -> Json:
    """Seal source and inputs before validation can change the candidate's receipts."""
    wall = datetime.now(UTC).isoformat()
    progress("preconditions_and_independent_admission", 0, 3)
    with (
        patch.object(old, "authenticate", authenticate),
        patch.object(old, "OWNED", OWNED),
        patch.object(old, "TEST", TEST),
    ):
        work = old.measure(root, raw, private, run_date)
    work.update(started_at_utc=wall, ended_at_utc=datetime.now(UTC).isoformat())
    work["code_config_hashes"] += [
        reference(ROOT / p)
        for p in [
            "docs/research-notes/v745-hardware-operation-boundary.md",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "openspec/capabilities/hardware-operation-boundary/spec.md",
            "python/carnot/reporting/hardware_operation_boundary_8679.py",
            "python/carnot/reporting/hardware_boundary_costs_8679.py",
            "python/carnot/reporting/hardware_evidence_validation_8665.py",
            "python/carnot/reporting/hardware_evidence_runner_8651.py",
            "python/carnot/reporting/spline_precision_study_8692.py",
            "research-references.md",
            "research-hardware-wishlist.md",
            "ops/north-star.md",
        ]
    ]
    atomic_json(raw / "measurement.json", work)
    atomic_json(raw / "measurement.seal.json", reference(raw / "measurement.json"))
    (raw / "measurement.json").chmod(0o444)
    (raw / "measurement.seal.json").chmod(0o444)
    progress("measurement_frozen", 3, 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Numerical readiness is a prerequisite, while complete-service speed stays unknown."""
    value = BASE_BUILD(work, receipts, raw, output)
    branches = work["evidence"]["branches"]
    numeric = branches[1]
    value.update(
        experiment_id=8693,
        task_id=TASK,
        milestone="2026.11.745",
        random_seed=7458693,
        honest_verdict=f"complete_{value['verdict_class']}_v745_hardware_operation_boundary",
        started_at_utc=work["started_at_utc"],
        ended_at_utc=(
            datetime.fromisoformat(work["started_at_utc"]) + timedelta(seconds=work["duration_s"])
        ).isoformat(),
        integer_precision_rows=numeric.get("integer_precision_rows", []),
        numerical_operation_inventory=numeric.get("operation_inventory", []),
        accumulator_bounds=numeric.get("accumulator_bounds", []),
        substrate_roles=dict(
            reader="aggregation_from_upstream_artifacts",
            controls="deterministic_verifier",
            numerical_head="verifier_ensemble_against_cached_candidates; upstream only",
        ),
    )
    value["rows"] = [
        dict(
            unit_id=f"exp{b['experiment_id']}",
            arm=b["boundary"],
            metric="qualified_original_evidence",
            value=int(b["qualified"]),
            status="completed" if b["qualified"] else "censored",
            missing_reason=b["missing_reason"],
        )
        for b in branches
    ]
    value["workload_boundary_inventory"] = [
        dict(
            boundary=b["boundary"],
            producer=b["experiment_id"],
            admitted=b["qualified"],
            complete_measured_cells=b["cells"],
            measured_host_work=b.get("operation_inventory", []),
            transferable_numerical_operations=[
                "basis",
                "multiply_divide",
                "checked_accumulate",
                "sparse_brier_write",
            ],
            host_work=[
                "sigmoid_logit",
                "coefficient_decay",
                "float64_fallback",
                "startup_shutdown",
            ],
            transport="retain original disjoint spans; absent operands remain unmeasured",
            durable_store="retain original storage spans; CPU micro-operations do not close the learning lifecycle",
            unsupported_functions=[
                "target_board_integer_spline",
                "distribution_sampling",
                "complete_on_chip_learning",
            ],
            missing_reason=b["missing_reason"],
        )
        for b in branches
    ]
    value["per_board_results"] = [
        dict(
            board=board,
            historical_scope=scope,
            mapping_ready_score=value["hardware_mapping_ready_score"],
            physical_ready_score=0,
            implemented_integer_spline=False,
            physical_execution_this_invocation=False,
            future_kernel="Nine-input 109-coefficient spline basis, integer residual and sparse Brier updates",
            accumulator="Checked64-bit products/sums; 32/24 worst-case37-product bound exceeds64 bits; observed ranges are finite-panel evidence",
            transfer="Typed9-feature input, prediction and delayed-label transport with disjoint full-service clocks",
            persistence="109 coefficients, lazy global decay, scaler/schema, durable atomic checkpoint and feedback drain",
            operator_prerequisite=prerequisite,
        )
        for board, scope, prerequisite in [
            (
                "KV260",
                "qualified quadratic Ising k_max<=5",
                "Compatible implemented kernel and parity; future SSH/scp/xmutil access",
            ),
            (
                "PolarFire",
                "Linux CPU dispatch only",
                "Fabric kernel, numerical parity and separately measured CPU/fabric transport",
            ),
            (
                "GateMate",
                "physical0xffffffff blocker",
                "Dated operator physical change; IDCODE0x20000001; ordered n16 smoke",
            ),
        ]
    ]
    value["complete_boundary_rows"] += [
        dict(
            boundary="integer_precision",
            producer=8692,
            measured_host_work=numeric.get("operation_inventory", []),
            transferable_numerical_operations=[
                "fixed_point_basis",
                "integer_residual",
                "integer_brier_sparse_write",
            ],
            host_required=[
                "sigmoid",
                "logit",
                "coefficient_decay",
                "float64_fallback",
                "state_copy",
            ],
            transport=None,
            durable_store="host_durable_storage; measured CPU micro-operation only",
            unsupported_functions=[
                "physical_spline_kernel",
                "distribution_sampler",
                "complete_on_chip_learner",
            ],
            total_s=None,
            compatible_fraction=None,
            conditional_ceiling=None,
            missing_reason="CPU_micro_spans_are_not_complete_service_or_physical_latency",
        )
    ]
    value["missing_cost_rows"].append(
        dict(
            producer=8692,
            missing_reason="No implemented compatible spline; no complete transfer and learning interval",
        )
    )
    value["missing_cost_rows"].append(
        dict(
            producer=8692,
            missing_reason="coefficient_decay_cost_not_isolated_from_sparse_update; host_fallback_and_nonlinear_costs_retained",
        )
    )
    value["conditional_acceleration_bounds"] = [
        dict(
            producer=b["experiment_id"],
            boundary=b["boundary"],
            cohort=c.get("cohort"),
            arm=c.get("arm"),
            total_s=c.get("total_s"),
            compatible_fraction=c.get("compatible_fraction"),
            conditional_ceiling=c.get("conditional_ceiling"),
            achieved_speedup=None,
            missing_reason=c.get(
                "missing_reason",
                b["missing_reason"] or "missing_compatible_operation_implementation",
            ),
            interpretation="Conditional1/(1-f) only for disjoint measured compatible implemented operations",
        )
        for b in branches
        for c in (b["cells"] or [{}])
    ]
    value["conditional_ceiling_rows"] = value["conditional_acceleration_bounds"]
    value["branch_gate_values"].update(integer_precision_admitted_score=int(numeric["qualified"]))
    value["integer_precision_admitted_score"] = int(numeric["qualified"])
    value["operation_support_rows"] += [
        dict(
            row,
            evidence_scope="CPU emulation; resource/range only",
            implemented_on_any_target_board=False,
        )
        for row in numeric.get("operation_inventory", [])
    ]
    value["preserved_quarantines"]["Exp8667"] = dict(
        flagged_adversarial=True,
        verdict_class="disqualified",
        reason="Historical nested diagnostic mutation failed owned replay and coverage; later code fix does not qualify original bytes",
    )
    value["preserved_historical_outcomes"] += [
        "V744 ended fourteen scheduling slots, not fourteen successful studies. Learning, relational controls and batching were skipped.",
        "Exp8677 qualified single-owner observations; Exp8678 closed its fixed panel with zero complete horizons; Exp8679 mapped without physical acceleration.",
        "Preserve V733 source-information, V735 cold-cost, V737 supervisor and V743 batch-one nulls.",
    ]
    value["hardware_100x_requirements"].update(
        integer_range_evidence_admitted=numeric["qualified"],
        generic_32bit_accumulation_proven=False,
        missing_conditions=[
            "Compatible implemented kernel and general input-range/accumulator proof",
            "Complete causal learning, retention, transfer, host sigmoid/logit, coefficient decay and fallback costs",
            "Durable updates and feedback drain; unchanged typed decisions",
            "Unaccelerated share strictly below0.01 plus measured finite kernel gain sufficient for100x",
        ],
    )
    value["literature_operation_mapping"] += [
        dict(
            source="https://arxiv.org/abs/2602.02056v4",
            operation="sparse fixed-point B-spline updates",
            local_limit="Motivates at most37 local writes; host nonlinear, decay and durable store remain. No paper ratios transferred.",
        ),
        dict(
            source="https://arxiv.org/abs/2512.12850v3",
            operation="LUT discretized spline evaluation",
            local_limit="Needs coefficient/LUT update coherence and quantization parity for this head.",
        ),
        dict(
            source="https://arxiv.org/abs/2610.31339v1",
            operation="finite-precision conditional energy sampling",
            local_limit="Abstract-level comparison only; needs random source, conditional sampler and distribution tests. Quadratic Ising and spline evaluation do not implement it.",
        ),
    ]
    value["methodology_note"] = (
        "Authenticate and aggregate upstream integer/resource rows and complete-cost primitives. No model load, head evaluation, benchmark or device probe. Original producer construction and terminal validation authenticate numerical checkpoints separately. CPU micro-operation spans are not physical or complete-service timings."
    )
    value["raw_shard_hashes"].append(reference(raw / "measurement.seal.json"))
    value["acceptance_gate_principles"] = {
        k: "Original execution completeness is distinct from numerical, physical and scientific benefit readiness."
        for k in value["acceptance_gates"]
    }
    value["gate_check_summary"] = [
        dict(row, operator=row.get("operator", row.get("op", "==")))
        for row in value["gate_check_summary"]
    ]
    value["field_principles"] = {
        k: "Invocation-bound original bytes; missing differs from zero; mapping, numerical and physical readiness remain separate."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold construction reauthenticates original producers beyond the candidate's hashes."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(checked(value["work_reference"]).read_bytes())
        raw = Path(value["work_reference"]["path"]).parent
        measured = json.loads(
            checked(json.loads((raw / "measurement.seal.json").read_bytes())).read_bytes()
        )
        if any(work[k] != measured[k] for k in ["started_at_utc", "ended_at_utc"]):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    with patch.object(old, "authenticate", authenticate), patch.object(old, "build", build):
        return old.replay(path)
