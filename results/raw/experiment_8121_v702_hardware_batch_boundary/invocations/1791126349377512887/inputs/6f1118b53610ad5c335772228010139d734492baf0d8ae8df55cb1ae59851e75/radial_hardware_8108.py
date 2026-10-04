"""REQ-VERIFY-8108, REQ-REPORT-8108: CPU storage precision is not fabric speed."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from carnot.reporting import hardware_feature_8068 as custody
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import radial_memory_8085 as kernel

Json = dict[str, Any]
ROOT = custody.ROOT
CONTRACTS = {
    "KV260": ("fpga_fabric", 5, None),
    "PolarFire": ("linux_cpu", None, None),
    "GateMate": ("none", None, "0xffffffff"),
}
RESOURCES = [
    *custody.RESOURCES,
    "openspec/capabilities/verification/spec.md",
    "python/carnot/reporting/hardware_workload_8081.py",
    "scripts/experiments/experiment_8081_v699_hardware_workload_boundary.py",
    "ops/north-star.md",
]
COMPONENTS = {
    "scoring_ns",
    "judgment_lookup_ns",
    "threshold_fallback_ns",
    "missing_judgment_fallback_ns",
    "durable_commit_ns",
    "loading_ns",
    "rendering_ns",
    "extraction_ns",
    "hashing_ns",
    "storage_ns",
    "invalidation_ns",
    "lifecycle_ns",
    "update_serialization_ns",
}
CONFIG: Json = dict(
    seed=7008085,
    systems=64,
    bits=[8, 12, 16],
    center_range=[-4, 4],
    coefficient_range=[-8, 8],
    scaled_input_range=[-4, 4],
    sigma_range=[0.25, 16],
    arithmetic="float64; quantized storage only",
    thresholds=[0.1, 0.5],
    floating_allowance=1e-11,
    transcendental_assumption="exp and sigmoid at most 4 ulp on this host",
)


def numerical(system: Json, bits: int) -> Json:
    """Bound storage error using Gaussian and sigmoid slopes, never fitted widths.

    The Gaussian gradient norm is at most exp(-1/2)/sigma; sigmoid slope is
    at most 1/4. A 1e-11 allowance dominates float64 operation rounding for
    d=9, m<=28 and the declared domain under the stated 4 ulp assumption.
    This envelope describes host arithmetic, not an unimplemented fixed-point LUT.
    """
    state = system["state"]
    x = kernel.matrix(system["x"])
    g = state["geometry"]
    sigma = float(g["sigma"])
    centers = np.asarray([c["x"] for c in state["centers"]], dtype=float)
    theta = np.asarray(state["coefficients"], dtype=float)
    if (
        bits not in CONFIG["bits"]
        or not 0.25 <= sigma <= 16
        or not 1 <= len(centers) <= 28
        or centers.shape[1:] != (9,)
        or theta.shape != (len(centers) + 1,)
        or not np.isfinite(centers).all()
        or not np.isfinite(theta).all()
    ):
        raise ValueError("precision_domain")
    z = (x - np.asarray(g["mean"])) / np.asarray(g["std"])
    if not np.isfinite(z).all():
        raise ValueError("scaled_input_domain")
    levels = 2**bits - 1
    cstep, tstep = 8 / levels, 16 / levels
    qc = np.rint((np.clip(centers, -4, 4) + 4) / cstep) * cstep - 4
    qt = np.rint((np.clip(theta, -8, 8) + 8) / tstep) * tstep - 8
    clip = int(np.count_nonzero(abs(centers) > 4) + np.count_nonzero(abs(theta) > 8))
    reference = kernel.predict(state, x)
    phi = np.exp(-np.sum((z[:, None, :] - qc[None, :, :]) ** 2, axis=2) / (2 * sigma**2))
    quantized = expit(qt[0] + phi @ qt[1:])
    # Clipped operands use their actual residual; they never earn domain credit.
    cr = max(cstep / 2, float(np.max(abs(centers - qc))))
    tr = max(tstep / 2, float(np.max(abs(theta - qt))))
    logit_bound = len(theta) * tr + float(np.sum(abs(qt[1:]))) * math.exp(-0.5) / sigma * 3 * cr
    bound = float(np.nextafter(min(1.0, logit_bound / 4 + CONFIG["floating_allowance"]), math.inf))
    error = abs(reference - quantized)
    outside = np.any(abs(z) > 4, axis=1)
    near = np.minimum(abs(quantized - 0.1), abs(quantized - 0.5)) <= bound + 1e-12
    flags = near | outside | bool(clip)
    issued = np.where(flags, reference, quantized)
    actions = [kernel.action(float(p)) for p in issued]
    canonical = [kernel.action(float(p)) for p in reference]
    return dict(
        source_id="supplied_numerical_fixture",
        unit_id=f"seed{system['seed']}-b{bits}",
        arm=f"quantized_{bits}",
        condition="storage_precision_float64_arithmetic",
        issued_state=canonical_hash(state),
        metric="probability_error",
        numerator=float(np.max(error)),
        denominator=1,
        status="completed",
        exclusion_reason=None,
        bits=bits,
        points=len(x),
        clipping_count=clip,
        outside_domain_count=int(outside.sum()),
        reference_probabilities=reference.tolist(),
        quantized_probabilities=quantized.tolist(),
        issued_probabilities=issued.tolist(),
        fallback_flags=flags.tolist(),
        issued_actions=actions,
        reference_actions=canonical,
        probability_error_bound=bound,
        maximum_probability_error=float(np.max(error)),
        unguarded_action_disagreements=sum(
            kernel.action(float(p)) != a for p, a in zip(quantized, canonical, strict=True)
        ),
        action_disagreements=sum(a != b for a, b in zip(actions, canonical, strict=True)),
        center_step=cstep,
        coefficient_step=tstep,
    )


def summarize(rows: list[Json]) -> Json:
    """Reduce individual probabilities so a forged maximum cannot hide a violation."""
    valid = all(
        max(
            abs(a - b)
            for a, b in zip(r["reference_probabilities"], r["quantized_probabilities"], strict=True)
        )
        <= r["probability_error_bound"]
        and r["action_disagreements"] == 0
        for r in rows
    )
    points = sum(r["points"] for r in rows)
    return dict(
        passed=valid,
        fallback_fraction=sum(sum(r["fallback_flags"]) for r in rows) / points if points else 0,
        maximum_probability_error=max((r["maximum_probability_error"] for r in rows), default=0),
    )


def authenticate(path: Path, eid: int, field: str, raw: Path, data: Json) -> tuple[Json, bool]:
    """Only byte-bound terminal success grants use of a particular readiness field."""
    begin = len(data["checks"])
    saved = custody.seal(path, None, raw, data, f"exp{eid}")
    if saved is None:
        return {}, False
    value: Json = {}
    try:
        value = json.loads(saved.read_bytes())
        for key, wanted in [
            ("experiment_id", eid),
            (field, 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            data["checks"].append(custody.operand(f"exp{eid}", path, key, wanted, value.get(key)))
        terminal = Path(value["terminal_validation_sidecar_path"])
        binding = json.loads(terminal.read_bytes())["publication"]
        side = Path(binding["sidecar_path"])
        report = read_bound_sidecar(path, side)
        for key, wanted, observed in [
            ("publication.primary_sha256", sha256_file(path), binding["primary_sha256"]),
            ("publication.primary_path", str(path), binding["primary_path"]),
            ("report.passed", True, report["report"]["passed"]),
        ]:
            data["checks"].append(custody.operand(f"exp{eid}", path, key, wanted, observed))
        for p in (terminal, side):
            custody.seal(p, None, raw, data, f"exp{eid}")
    except (OSError, ValueError, KeyError) as exc:
        data["checks"].append(
            custody.operand(f"exp{eid}", path, "terminal_contract", "byte-bound pass", str(exc))
        )
    return value, all(r["passed"] for r in data["checks"][begin:])


def load(root: Path, raw: Path) -> Json:
    """Snapshot each board separately and admit only hashed numerical/timing shards."""
    data: Json = dict(systems=[], boards=[], costs=[], checks=[], references=[])
    for resource in RESOURCES:
        custody.seal(root / resource, None, raw, data, "local_resource")
    prior, qualified = authenticate(
        root / "results/experiment_8081_v699_hardware_workload_boundary.json",
        8081,
        "hardware_custody_ready_score",
        raw,
        data,
    )
    originals = {b["board"]: b for b in prior.get("board_rows", [])}
    for name, contract in CONTRACTS.items():
        begin = len(data["checks"])
        board = deepcopy(
            originals.get(
                name, dict(board=name, source_path=name, source_hash=None, custody_valid=False)
            )
        )
        source = root / board["source_path"]
        saved = custody.seal(source, board["source_hash"], raw, data, name)
        receipt = json.loads(saved.read_bytes()) if saved else {}
        transcript = receipt.get("kv260_terminal_transcript_path") or receipt.get(
            "raw_dispatch_transcript_path"
        )
        digest = receipt.get("kv260_terminal_transcript_sha256") or next(
            (
                r.get("latest_receipt_hash")
                for r in receipt.get("board_rows", [])
                if r.get("board") == name
            ),
            None,
        )
        board["original_receipt"] = receipt
        if transcript and digest:
            transcript_path = root / transcript
            saved_transcript = custody.seal(
                transcript_path,
                digest if digest.startswith("sha256:") else "sha256:" + digest,
                raw,
                data,
                name,
            )
            board["original_transcript"] = (
                json.loads(saved_transcript.read_bytes()) if saved_transcript else {}
            )
        observed = (board.get("processor_class"), board.get("k_max"), board.get("blocker"))
        data["checks"].append(
            custody.operand(
                name, source, "recorded_workload_boundary", list(contract), list(observed)
            )
        )
        board["custody_valid"] = qualified and all(r["passed"] for r in data["checks"][begin:])
        data["boards"].append(board)
        print(
            f"[exp8108] custody completed={len(data['boards'])} pending={3 - len(data['boards'])}",
            flush=True,
        )
    for eid, label, field, shard, key, target in [
        (
            8085,
            "v700_radial_memory_kernel",
            "kernel_ready_score",
            "evidence.json",
            "systems",
            "systems",
        ),
        (
            8106,
            "v701_radial_service_cost",
            "service_cost_ready_score",
            "primitive_rows.json",
            "paired_service_rows",
            "costs",
        ),
    ]:
        value, valid = authenticate(
            root / f"results/experiment_{eid}_{label}.json", eid, field, raw, data
        )
        if valid:
            ref = next(
                (
                    r
                    for r in value.get("raw_shard_hashes", [])
                    if Path(r["path"]).name == shard and "failed_attempts" not in r["path"]
                ),
                {},
            )
            saved = custody.seal(
                root / ref.get("path", shard), ref.get("sha256"), raw, data, f"exp{eid}"
            )
            if saved:
                payload = json.loads(saved.read_bytes())
                data[target] = payload.get(key, [])
    return data


def service(pairs: list[Json]) -> Json:
    """Charge the complete interval; unknown board transport can only reduce this ceiling."""
    rows: list[Json] = []
    seen: set[tuple[str, str]] = set()
    try:
        for pair in pairs:
            if {a["arm"] for a in pair["arms"]} != {"python", "rust"} or len(pair["arms"]) != 2:
                raise ValueError("paired_arms")
            for a in pair["arms"]:
                identity = (pair["unit_id"], a["arm"])
                c = a["components"]
                total = a.get("total_ns", a.get("transaction_ns"))
                if (
                    identity in seen
                    or not COMPONENTS <= set(c)
                    or any(type(v) is not int or v < 0 for v in c.values())
                    or type(total) is not int
                    or total <= sum(c.values())
                ):
                    raise ValueError("primitive_cost_contract:" + str(identity))
                seen.add(identity)
                arithmetic = c["scoring_ns"]
                rows.append(
                    dict(
                        unit_id=pair["unit_id"],
                        source_id=pair.get("source_id", a.get("source_id")),
                        arm=a["arm"],
                        condition=pair.get("mode", a.get("condition")),
                        centers=a.get("centers", pair.get("centers")),
                        total_ns=total,
                        arithmetic_ns=arithmetic,
                        nonarithmetic_ns=total - arithmetic,
                        components=c,
                        uninstrumented_ns=total - sum(c.values()),
                        transport_ns=None,
                        conditional_speedup_bound=total / (total - arithmetic),
                        transport_assumption="zero added board transport: optimistic upper ceiling",
                    )
                )
        if not rows:
            raise ValueError("no_authenticated_service_primitives")
    except (KeyError, ValueError, TypeError) as exc:
        return dict(
            status="unavailable",
            failed_gate=str(exc),
            rows=[],
            arithmetic_fraction=None,
            conditional_speedup_bound=None,
        )
    total = sum(r["total_ns"] for r in rows)
    arithmetic = sum(r["arithmetic_ns"] for r in rows)
    return dict(
        status="available",
        rows=rows,
        arithmetic_fraction=arithmetic / total,
        conditional_speedup_bound=total / (total - arithmetic),
        maximum_cell_speedup_bound=max(r["conditional_speedup_bound"] for r in rows),
    )


def reduce(data: Json) -> Json:
    """A complete assessment can preserve terminal external blocks without retrying."""
    boards = deepcopy(data["boards"])
    for b in boards:
        b["custody_valid"] = bool(
            b["custody_valid"]
            and (b.get("processor_class"), b.get("k_max"), b.get("blocker"))
            == CONTRACTS[b["board"]]
        )
        b.update(
            status="completed" if b["custody_valid"] else "blocked",
            current_hardware_execution=False,
            radial_mapping_implemented=False,
        )
    precision = []
    print(f"[exp8108] before_cpu_benchmark completed=0 pending={len(data['systems'])}", flush=True)
    for i, system in enumerate(data["systems"]):
        precision.extend(numerical(system, bits) for bits in CONFIG["bits"])
        if (i + 1) % 16 == 0:
            print(
                f"[exp8108] cpu completed={i + 1} pending={len(data['systems']) - i - 1}",
                flush=True,
            )
    print(f"[exp8108] after_cpu_benchmark completed={len(data['systems'])} pending=0", flush=True)
    summary = summarize(precision)
    costs = service(data["costs"])
    checks = deepcopy(data["checks"])
    checks.append(
        custody.operand(
            "exp8085", Path("frozen_systems"), "fixture_count", 64, len(data["systems"])
        )
    )
    checks.append(
        custody.operand(
            "exp8106",
            Path("frozen_service_primitives"),
            "primitive_cost_contract",
            "available",
            costs["status"],
        )
    )
    scoped = (
        len(data["systems"]) == 64 and len(boards) == 3 and all(b["custody_valid"] for b in boards)
    )
    ready = int(scoped and summary["passed"])
    resource = "radial_fixtures" if len(data["systems"]) != 64 else "board_evidence"
    verdict = "complete_null_radial_hardware_boundary" if ready else "complete_blocked_" + resource
    klass = "null" if ready else "blocked"
    if not summary["passed"]:
        verdict, klass = "complete_disqualified_precision_bound", "disqualified"
    return dict(
        honest_verdict=verdict,
        verdict_class=klass,
        hardware_boundary_ready_score=ready,
        hardware_execution=False,
        board_rows=boards,
        precision_rows=precision,
        rows=precision,
        precision_summary=summary,
        fallback_fraction=summary["fallback_fraction"],
        arithmetic_fraction=costs["arithmetic_fraction"],
        conditional_speedup_bound=costs["conditional_speedup_bound"],
        service_cost_subresult=costs,
        gate_check_summary=checks,
        preconditions_checked=True,
        verifier_is_oracle=False,
        exposure_scope="supplied fixtures and exposed historical evidence",
        generalized_learning_benefit_score=0,
        trained_head_specs=[],
        intended_count=192,
        eligible_count=len(precision),
        independent_count=0,
        completed_count=len(precision),
        excluded_count=192 - len(precision),
        censored_count=0,
        failed_count=0 if summary["passed"] else 1,
        sample_size_budget=dict(
            seeded_systems=64, precisions=3, natural_sources=0, seeds_are_not_independent=True
        ),
        config=CONFIG,
        acceptance_gates=dict(
            scoped_assessment=bool(ready),
            numerical_envelope=summary["passed"],
            service_join_optional=costs["status"] == "available",
        ),
        missing_mapping_operations=[
            "scaled radial distance",
            "Gaussian exp/LUT error contract",
            "center/coefficient memory and version transfer",
            "sigmoid",
            "typed threshold fallback",
            "feedback gradient/update and durable state synchronization",
            "measured board transport",
        ],
        learning_path=dict(
            gpu="Batched distance/features and gradients; measure transfer and batch size.",
            rust="Reuse the qualified Rust CPU kernel; this run does not load a native binding.",
            fpga="Future distance/LUT arithmetic needs new mapping, bounded numerical error and custody.",
            interfaces="Existing quadratic Ising fabric and TSU interfaces cannot accept radial centers unchanged.",
        ),
        research_program_100x=dict(
            target=100,
            measured_arithmetic_free_upper_bound=costs["conditional_speedup_bound"],
            necessary_arithmetic_fraction=0.99,
            promise=False,
            missing="Device mapping, transport and complete service benchmark; host/acquisition overhead remains.",
        ),
        observed_domain_panel=None,
    )
