"""REQ-REPORT-8055: numerical guard parity and board custody are separate claims."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.experiment_8053_v697_guarded_transaction_cost import workloads
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.verify import feedback_constrained_8051 as learning

Json = dict[str, Any]
Array = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[3]
CONFIG: Json = dict(
    seed=8055,
    numerical_budget_s=600,
    fractional_bits=12,
    state_bits=24,
    accumulator_bits=32,
    alphas=[1, 0.5, 0.25, 0.125, 0],
    rounding_radius=1 / 8192,
    host_allowance=2**-40,
    interval_method="analytical input cells and operation rounding; no fitted widths",
    state_commit="float64 authoritative shadow; staged integer witness",
    statistics="descriptive exposed development; no independent benefit test",
)
SOURCES = {
    8042: ("experiment_8042_v696_precision_fallback_boundary", "hardware_custody_ready_score"),
    8051: ("experiment_8051_v697_feedback_constrained_learning", "learning_trajectory_ready_score"),
    8053: ("experiment_8053_v697_guarded_transaction_cost", "native_transaction_ready_score"),
}


def predicate(bounds: list[float]) -> bool | None:
    """Only an interval wholly on one side certifies a strict guard comparison."""
    return True if bounds[0] > 0 else False if bounds[1] <= 0 else None


def scan(
    head: Json,
    theta: Array,
    x: Array,
    y: Array,
    radius: float,
    *,
    staged: Array | None = None,
    input_overflow: bool = False,
) -> Json:
    """Propagate fixed input cells through products, sums, calibration and metrics.

    The signed32 prefix bound detects overflow before commit. Saturation uses the
    full probability range, so no observed error selects an interval width.
    """
    if not all(np.isfinite(v).all() for v in (theta, x, y)):
        raise ValueError("nonfinite_operand")
    r = CONFIG["rounding_radius"]
    qt = np.clip(np.rint(theta * 4096), -(2**23), 2**23 - 1) if staged is None else staged
    qx = np.clip(np.rint(x * 4096), -(2**23), 2**23 - 1)
    products = np.rint(qx * qt / 4096)
    totals = products.sum(axis=1)
    a, b = head["calibration"]
    qa, qb = np.clip(np.rint(np.array([a, b]) * 4096), -(2**23), 2**23 - 1)
    z = (qa + np.rint(qb * totals / 4096)) / 4096
    overflow = bool(
        input_overflow
        or np.any(np.abs(theta) >= 2048)
        or np.any(np.abs(x) >= 2048)
        or max(abs(a), abs(b)) >= 2048
        or np.any(np.abs(products).sum(axis=1) >= 2**31)
        or np.any(np.abs(z * 4096) >= 2**31)
    )
    error = np.abs(x) @ np.full(len(theta), radius) + r * np.abs(theta).sum()
    error += len(theta) * (r * radius + r)
    error = abs(b) * error + r * (np.abs(totals / 4096) + error) + 2 * r
    error += CONFIG["host_allowance"] * (1 + np.abs(z)) * len(theta)
    lo, hi = expit(z - error), expit(z + error)
    if overflow:
        lo, hi = np.zeros(len(y)), np.ones(len(y))
    lo = np.maximum(0, lo - CONFIG["host_allowance"])
    hi = np.minimum(1, hi + CONFIG["host_allowance"])
    floating = expit(a + b * (x @ theta))
    quantized = expit(z)
    possible = []
    for lower, upper in zip(lo, hi, strict=True):
        acts = {learning.action(float(lower)), learning.action(float(upper))}
        if lower <= 0.1 <= upper or lower <= 0.5 <= upper:
            acts.add("escalate")
        possible.append(sorted(acts))
    costs = [
        [float(learning.loss(act, int(label))) for act in acts]
        for acts, label in zip(possible, y, strict=True)
    ]
    squared_lo = np.where(y == 0, lo**2, (1 - hi) ** 2)
    squared_hi = np.where(y == 0, hi**2, (1 - lo) ** 2)
    return dict(
        probability_bounds=np.column_stack((lo, hi)).tolist(),
        contained=bool(np.all((lo <= floating) & (floating <= hi))),
        brier_bounds=[float(squared_lo.mean()), float(squared_hi.mean())],
        cost_bounds=[
            float(np.mean([min(c) for c in costs])),
            float(np.mean([max(c) for c in costs])),
        ],
        possible_actions=possible,
        overflow=overflow,
        prediction_fallbacks=sum(len(acts) > 1 or overflow for acts in possible),
        prediction_count=len(y),
        quantized_probabilities=quantized.tolist(),
    )


def numeric(data: Json, *, budget_s: float = 600) -> Json:
    """Replay released guard buffers and select float64 before committing uncertainty."""
    began = time.monotonic()
    rows, intervals, fallbacks, predictions, checkpoints = [], [], [], [], {}
    head = data["head"]
    initial = learning.old.coefficients(head)
    vectors = {
        i: learning.old.design(head, source)
        for i, source in enumerate(data.get("sources", []))
        if source["public_eligible"]
    }
    for index, case in enumerate(data["cases"]):
        if time.monotonic() - began > budget_s:
            raise TimeoutError("numerical_budget_s")
        theta, delta = np.asarray(case["before"]), np.asarray(case["delta"])
        x = (
            np.asarray(case["guard_design"])
            if "guard_design" in case
            else np.asarray([vectors[i] for i, _ in case["guards"]])
        )
        y = np.asarray([label for _, label in case["guards"]], dtype=float)
        if not len(y):
            x = np.empty((0, len(theta)))
        reference_result = learning.guard(head, theta, delta, initial, x, y, case["arm"])
        ready = bool(reference_result["diagnostics"])
        uncertain = overflow = False
        admissible = []
        before = time.perf_counter_ns()
        if ready:
            baseline = scan(head, initial, x, y, CONFIG["rounding_radius"])
            for diagnostic in reference_result["diagnostics"]:
                alpha = diagnostic["alpha"]
                encoded_theta = np.clip(np.rint(theta * 4096), -(2**23), 2**23 - 1)
                encoded_delta = np.clip(np.rint(delta * 4096), -(2**23), 2**23 - 1)
                product = np.rint(encoded_delta * round(alpha * 4096) / 4096)
                proposal = encoded_theta + product
                staged = np.clip(proposal, -(2**23), 2**23 - 1)
                input_overflow = bool(
                    np.any(np.abs(delta) >= 2048)
                    or np.any(proposal < -(2**23))
                    or np.any(proposal >= 2**23)
                )
                candidate = scan(
                    head,
                    theta + alpha * delta,
                    x,
                    y,
                    CONFIG["rounding_radius"] * (3 + abs(alpha)),
                    staged=staged,
                    input_overflow=input_overflow,
                )
                states = []
                for name, field in [("brier", "brier_bounds"), ("typed_cost", "cost_bounds")]:
                    c, b = candidate[field], baseline[field]
                    states.append(predicate([c[0] - b[1] - 1e-12, c[1] - b[0] - 1e-12]))
                definite = any(
                    c == ["accept"] and "accept" not in b and label == 1
                    for c, b, label in zip(
                        candidate["possible_actions"], baseline["possible_actions"], y, strict=True
                    )
                )
                possible = any(
                    "accept" in c and b != ["accept"] and label == 1
                    for c, b, label in zip(
                        candidate["possible_actions"], baseline["possible_actions"], y, strict=True
                    )
                )
                states.append(True if definite else None if possible else False)
                overflow = overflow or candidate["overflow"] or baseline["overflow"]
                uncertain = (
                    uncertain
                    or None in states
                    or overflow
                    or candidate["prediction_fallbacks"] > 0
                    or baseline["prediction_fallbacks"] > 0
                )
                admissible.append(not any(states))
                contained = candidate["contained"] and baseline["contained"]
                for operand, actual in [
                    (candidate, diagnostic["candidate"]),
                    (baseline, diagnostic["baseline"]),
                ]:
                    contained = contained and all(
                        operand[field][0] - 1e-12 <= actual[key] <= operand[field][1] + 1e-12
                        for key, field in [("brier", "brier_bounds"), ("typed_cost", "cost_bounds")]
                    )
                intervals.append(
                    dict(
                        identity=case["identity"],
                        alpha=alpha,
                        contained=contained,
                        baseline={
                            k: v
                            for k, v in baseline.items()
                            if k
                            not in {
                                "probability_bounds",
                                "possible_actions",
                                "quantized_probabilities",
                            }
                        },
                        candidate={
                            k: v
                            for k, v in candidate.items()
                            if k
                            not in {
                                "probability_bounds",
                                "possible_actions",
                                "quantized_probabilities",
                            }
                        },
                        probability_bound_checksum=canonical_hash(
                            [baseline["probability_bounds"], candidate["probability_bounds"]]
                        ),
                        predicate_states=states,
                        predicate_names=["brier", "typed_cost", "new_false_accept"],
                        numerator=len(y),
                        denominator=len(y),
                    )
                )
                predictions.append(
                    dict(
                        identity=case["identity"],
                        alpha=alpha,
                        numerator=candidate["prediction_fallbacks"],
                        denominator=len(y),
                    )
                )
        passing = [a for a, good in zip(CONFIG["alphas"], admissible, strict=ready) if good]
        chosen = (
            1
            if case["arm"] == "unconstrained"
            else passing[0]
            if passing
            else 0
            if not ready
            else None
        )
        if uncertain:
            chosen = reference_result["alpha"]
        committed = (initial if chosen is None else theta + chosen * delta).tolist()
        checkpoint = dict(
            parameters=committed,
            alpha=chosen,
            reset=chosen is None,
            rejected=chosen != 1,
            status="waiting_guard" if not ready else "reset" if chosen is None else "commit",
        )
        restarted = json.loads(json.dumps(checkpoint))
        expected = case.get("expected", reference_result)
        passed = all(checkpoint[k] == reference_result[k] == expected[k] for k in checkpoint)
        rows.append(
            dict(
                identity=case["identity"],
                arm=case["arm"],
                seed=case["seed"],
                natural=case["natural"],
                alpha=chosen,
                passed=passed,
                restart_agrees=restarted == checkpoint,
                numerator=int(passed),
                denominator=1,
                independent=0,
            )
        )
        fallbacks.append(
            dict(
                identity=case["identity"],
                used_float64=uncertain,
                arm=case["arm"],
                transaction_class=case.get("class", "synthetic"),
                overflow=overflow,
                unhandled_overflow=0,
                numerator=int(uncertain),
                denominator=1,
                guard_scan_and_fallback_ns=time.perf_counter_ns() - before,
            )
        )
        checkpoints[f"{case['arm']}/{case['seed']}"] = checkpoint
        if index % 100 == 0:
            print(
                f"[exp8055] replay elapsed_s={time.monotonic() - began:.3f} "
                f"completed={index + 1} pending={len(data['cases']) - index - 1}",
                flush=True,
            )
    return dict(
        rows=rows,
        interval_containment_rows=intervals,
        guard_fallback_rows=fallbacks,
        acceptance_parity_rows=rows,
        prediction_fallback_rows=predictions,
        final_checkpoints=checkpoints,
    )


def costs(
    rows: list[Json], fallback_fraction: float, overheads: dict[str, float] | None = None
) -> list[Json]:
    """Only the compatible fraction moves; storage and guard fallback remain serial."""
    result = []
    for row in rows:
        if row["excluded"] or not row["natural"]:
            continue
        total = row["transaction_ns"]
        components = row["components"]
        gradient = components["gradient_arithmetic_ns"]
        guard = components["guard_scans_ns"]
        compatible = gradient + guard * (1 - fallback_fraction)
        if not 0 <= compatible <= total or total <= 0:
            raise ValueError("cost_partition")
        overhead = (overheads or {}).get(row["condition"] + "/" + row["transaction_class"], 0.0)
        augmented = total + overhead
        result.append(
            dict(
                arm=row["arm"],
                condition=row["condition"],
                repetition=row["repetition"],
                transaction_class=row["transaction_class"],
                transaction_ns=total,
                compatible_arithmetic_ns=compatible,
                guard_scan_ns=guard,
                fallback_serial_ns=guard * fallback_fraction,
                storage_fsync_ns=components["storage_fsync_ns"],
                interval_cpu_serial_ns=overhead,
                augmented_transaction_ns=augmented,
                speedup_bound=augmented / (augmented - compatible + compatible / 100),
                device_transfer_ns=None,
                scope="hypothetical compatible arithmetic100x; measured CPU transaction; unknown transfer excluded",
            )
        )
    return result


def load(root: Path, raw: Path) -> Json:
    """Authenticate current terminal bindings and original board bytes independently."""
    raw.mkdir(parents=True, exist_ok=True)
    data: Json = dict(boards=[], checks=[], references=[], cases=[], costs=[])
    values: dict[int, Json] = {}
    for label in [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/precision_fallback_8042.py",
        "python/carnot/experiment_8042_v696_precision_fallback_boundary.py",
        "research-hardware-wishlist.md",
        "ops/hardware-bringup-prep.md",
        "research-references.md",
        ".venv/bin/python",
        ".venv/bin/pytest",
        ".venv/bin/coverage",
        ".venv/bin/ruff",
        ".venv/bin/mypy",
    ]:
        path = root / label
        data["checks"].append(
            dict(
                upstream_id="local_resource",
                path=str(path),
                sha256=sha256_file(path) if path.is_file() else None,
                check_name="resource_exists",
                artifact_field=label,
                expected=True,
                observed=path.is_file(),
                passed=path.is_file(),
            )
        )

    def retain(path: Path) -> Json:
        ref = reference(path)
        target = raw / "inputs" / ref["sha256"].split(":")[-1] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
        saved = reference(target)
        checked(dict(path=str(target), sha256=ref["sha256"]))
        data["references"].append(saved)
        return saved

    for identity, (name, ready) in SOURCES.items():
        path = root / "results" / (name + ".json")
        try:
            value = json.loads(path.read_bytes())
            side = Path(value["terminal_validation_sidecar_path"])
            terminal = json.loads(side.read_bytes())
            binding = terminal.get("publication", terminal)
            validator = Path(binding["sidecar_path"])
            report = json.loads(validator.read_bytes())
            operands = [
                ("experiment_id", identity, value.get("experiment_id")),
                (ready, 1, value.get(ready)),
                ("flagged_adversarial", False, value.get("flagged_adversarial")),
                ("primary_sha256", sha256_file(path), binding["primary_sha256"]),
                ("report.primary_sha256", sha256_file(path), report["primary_sha256"]),
                ("primary_path", str(path), binding["primary_path"]),
                ("report.passed", True, report["report"]["passed"]),
            ]
            for key, expected, observed in operands:
                data["checks"].append(
                    dict(
                        upstream_id=f"exp{identity}",
                        path=str(path),
                        sha256=sha256_file(path),
                        check_name=key,
                        artifact_field=key,
                        expected=expected,
                        observed=observed,
                        passed=type(expected) is type(observed) and expected == observed,
                    )
                )
            retain(side)
            retain(validator)
            if identity != 8042:
                retain(path)
            if all(c["passed"] for c in data["checks"] if c["upstream_id"] == f"exp{identity}"):
                values[identity] = value
        except (OSError, ValueError, KeyError) as error:
            data["checks"].append(
                dict(
                    upstream_id=f"exp{identity}",
                    path=str(path),
                    sha256=None,
                    check_name="input_contract",
                    artifact_field="authenticated_terminal_input",
                    expected="present with terminal byte binding",
                    observed=str(error),
                    passed=False,
                )
            )
    for board in values.get(8042, {}).get("board_rows", []):
        board = deepcopy(board)
        path = root / board["source_path"]
        observed = sha256_file(path) if path.is_file() else None
        gate = dict(
            upstream_id=board["board"],
            path=str(path),
            sha256=observed,
            check_name="original_board_sha256",
            artifact_field="source_hash",
            expected=board["source_hash"],
            observed=observed,
            passed=observed == board["source_hash"],
        )
        data["checks"].append(gate)
        board.update(
            custody_valid=gate["passed"],
            custody_checked_date="20261003",
            current_hardware_execution=False,
        )
        if gate["passed"]:
            retain(path)
            receipt = json.loads(path.read_bytes())
            transcript = receipt.get("kv260_terminal_transcript_path") or receipt.get(
                "raw_dispatch_transcript_path"
            )
            if transcript:
                transcript_path = root / transcript
                digest = receipt.get("kv260_terminal_transcript_sha256") or next(
                    r["latest_receipt_hash"]
                    for r in receipt["board_rows"]
                    if r["board"] == "PolarFire"
                )
                expected = "sha256:" + digest.removeprefix("sha256:")
                observed = sha256_file(transcript_path) if transcript_path.is_file() else None
                transcript_gate = dict(
                    gate,
                    path=str(transcript_path),
                    sha256=observed,
                    check_name="terminal_transcript_sha256",
                    artifact_field="terminal_transcript",
                    expected=expected,
                    observed=observed,
                    passed=expected == observed,
                )
                data["checks"].append(transcript_gate)
                board["custody_valid"] = transcript_gate["passed"]
                if transcript_gate["passed"]:
                    retain(transcript_path)
        data["boards"].append(board)
    if 8051 in values:
        value = values[8051]
        trajectory = Path(value["trajectory_directory"])
        try:
            sealed = {
                ref["path"]: ref
                for ref in value["raw_shard_hashes"] + value["checkpoint_references"]
            }
            for path in [
                trajectory / "inputs.json",
                trajectory / "methods.json",
                *sorted(trajectory.glob("seed-*/ledger.sqlite")),
            ]:
                checked(sealed[str(path)])
                retain(path)
            data.update(json.loads((trajectory / "inputs.json").read_bytes()))
            data["cases"] = workloads(trajectory, data)
        except (OSError, ValueError, KeyError) as error:
            data["checks"].append(
                dict(
                    upstream_id="exp8051",
                    path=str(trajectory),
                    sha256=None,
                    check_name="trajectory_operands",
                    artifact_field="qualified_guard_trajectory",
                    expected="byte-bound inputs and original ledgers",
                    observed=str(error),
                    passed=False,
                )
            )
    if 8053 in values:
        for ref in values[8053]["raw_shard_hashes"]:
            if "/transactions/" in ref["path"]:
                retain(checked(ref))
        data["costs"] = values[8053]["rows"]
    return data


def controls() -> Json:
    """Destructive and boundary fixtures test mechanics without adding source support."""
    head = dict(parameters=[0.0, 0.0], decay_scale=1.0, calibration=[0.0, 1.0])
    cases = []
    for name, value, delta, size in [
        ("noop", 0.0, 0.0, 8),
        ("destructive", 0.0, 10.0, 8),
        ("reset", 10.0, 0.0, 8),
        ("insufficient_guard", 0.0, 1.0, 2),
        ("threshold", -2.197224577, 0.0, 8),
        ("overflow", 1e8, 0.0, 8),
    ]:
        cases.append(
            dict(
                identity="control/" + name,
                arm="feedback_constrained",
                seed=0,
                slot=0,
                natural=False,
                before=[value, 0.0],
                delta=[delta, 0.0],
                guards=[[0, i % 2] for i in range(size)],
                guard_design=[[1.0, 0.0]] * size,
            )
        )
    measured = numeric(dict(head=head, cases=cases))
    return dict(
        working=all(r["passed"] and r["restart_agrees"] for r in measured["rows"])
        and all(r["contained"] for r in measured["interval_containment_rows"]),
        rows=measured["rows"],
        fallback_rows=[
            {k: v for k, v in r.items() if k != "guard_scan_and_fallback_ns"}
            for r in measured["guard_fallback_rows"]
        ],
        independent_count=0,
        verifier_is_oracle=True,
        claim_class="circular_positive",
        scope="synthetic destructive, reset, guard support, threshold and overflow controls",
    )


def reduce(data: Json) -> Json:
    """External blocks are terminal; numeric parity never claims device performance."""
    checks = deepcopy(data["checks"])
    measured: Json = dict(
        rows=[],
        interval_containment_rows=[],
        guard_fallback_rows=[],
        acceptance_parity_rows=[],
        prediction_fallback_rows=[],
        final_checkpoints={},
    )
    failed_numeric = any(not r["passed"] for r in checks if r["upstream_id"] == "exp8051")
    if data["cases"] and not failed_numeric:
        measured = numeric(data, budget_s=CONFIG["numerical_budget_s"])
    else:
        checks.append(
            dict(
                upstream_id="exp8051",
                path="numeric_branch",
                sha256=None,
                check_name="qualified_current_guard_trajectory",
                artifact_field="qualified_current_guard_trajectory",
                expected="present",
                observed="MISSING_QUALIFIED_CURRENT_GUARD_TRAJECTORY",
                passed=False,
            )
        )
    rows = measured["rows"]
    custody = len(data["boards"]) == 3 and all(b["custody_valid"] for b in data["boards"])
    numeric_good = bool(rows) and all(r["passed"] and r["restart_agrees"] for r in rows)
    numeric_good = numeric_good and all(
        r["contained"] for r in measured["interval_containment_rows"]
    )
    fallback = sum(r["used_float64"] for r in measured["guard_fallback_rows"])
    fraction = fallback / len(rows) if rows else None
    cost_good = bool(data["costs"]) and all(
        r["passed"] for r in checks if r["upstream_id"] == "exp8053"
    )
    timings = data.get("cpu_guard_cost_rows", measured["guard_fallback_rows"])
    timing_groups = {r["arm"] + "/" + r["transaction_class"] for r in timings}
    overheads = {
        key: float(
            np.mean(
                [
                    r["guard_scan_and_fallback_ns"]
                    for r in timings
                    if r["arm"] + "/" + r["transaction_class"] == key
                ]
            )
        )
        for key in timing_groups
    }
    bounds = costs(data["costs"], fraction or 0, overheads) if cost_good and rows else []
    if not cost_good:
        checks.append(
            dict(
                upstream_id="exp8053",
                path="cost_branch",
                sha256=None,
                check_name="qualified_complete_cost_rows",
                artifact_field="qualified_complete_cost_rows",
                expected="present",
                observed="MISSING_QUALIFIED_COMPLETE_COST_ROWS",
                passed=False,
            )
        )
    blocked = not rows or any(not r["passed"] for r in checks)
    positive_controls = controls()
    numeric_good = numeric_good and positive_controls["working"]
    measured.update(
        gate_check_summary=checks,
        board_rows=data["boards"],
        numeric_branch_status="qualified" if rows else "blocked",
        hardware_custody_ready_score=int(custody),
        guard_fallback_ready_score=int(numeric_good),
        guard_fallback_fraction=fraction,
        guard_fallback_counts=dict(
            numerator=fallback,
            denominator=len(rows),
            unit="candidate transaction; any uncertain guard predicate",
        ),
        guard_predicate_fallback_counts=dict(
            numerator=sum(
                state is None
                for r in measured["interval_containment_rows"]
                for state in r["predicate_states"]
            ),
            denominator=3 * len(measured["interval_containment_rows"]),
            unit="Brier, typed cost and false-accept predicate per alpha",
        ),
        prediction_fallback_count=sum(r["numerator"] for r in measured["prediction_fallback_rows"]),
        prediction_fallback_denominator=sum(
            r["denominator"] for r in measured["prediction_fallback_rows"]
        ),
        acceleration_bounds=bounds,
        honest_verdict="complete_blocked_missing_operands"
        if blocked
        else "complete_null_guarded_hardware_boundary"
        if numeric_good
        else "complete_disqualified_guard_parity",
        verdict_class="blocked" if blocked else "null" if numeric_good else "disqualified",
        acceptance_gate_results=dict(
            custody=custody,
            numerical_parity=numeric_good,
            qualified_costs=cost_good,
            measured_device_benefit=False,
        ),
        intended_count=len(data["cases"]),
        eligible_count=len(rows),
        completed_count=len(rows),
        excluded_count=0,
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=0,
        independent_count=len(
            {
                data["sources"][slot]["source_cluster_id"]
                for case in data["cases"]
                for slot, _ in case["guards"]
            }
        )
        if data.get("sources") and rows
        else 0,
        prediction_fallback_scope="guard scan probability readouts; no new transformer inference",
        cost_excluded_rows=[r for r in data["costs"] if r["excluded"]],
        cost_sample_counts=dict(
            intended=len(data["costs"]),
            completed=sum(not r["excluded"] for r in data["costs"]),
            excluded=sum(r["excluded"] for r in data["costs"]),
        ),
        sample_size_budget=dict(
            candidates=len(data["cases"]),
            numerical_budget_s=600,
            independent_unit="original source; seeds and fixtures add zero",
        ),
        compatibility_map=dict(
            KV260="SSH quadratic Ising k_max<=5 only",
            PolarFire="Linux CPU only; no spline fabric",
            GateMate="physical JTAG0xffffffff block",
            NPU="no qualified execution",
            TSU="no local hardware evidence",
            transformer_prefill="existing Ising fabric incompatible",
            arbitrary_spline_gradients="existing Ising fabric incompatible",
        ),
        missing_cost_components=[
            "device transfer",
            "transformer prefill",
            "full service",
            "device storage and guard scan execution",
        ],
        current_device_execution_count=0,
        purchase_recommendation="defer: no useful measured device bottleneck",
        verifier_is_oracle=False,
        genuine_headroom=False,
        generalized_learning_benefit_score=0,
        positive_control_results=positive_controls,
    )
    return measured
