"""REQ-REPORT-8661: complete clocks compare equal work under frozen inputs.

These CPU costs describe one exposed numerical head and fixed request corpus.
Readiness does not establish independent calibration or hardware acceleration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.installed_lean_8633 import inputs as original_inputs, load, reference
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8661_v743_installed_complete_cost"
TASK = "exp8661-installed-complete-cost"
CLI = f"scripts/experiments/{NAME}.py"
MODEL_SPECS: list[Json] = []
OWNED = [
    f"python/carnot/reporting/installed_cost{part}_8661.py"
    for part in ["", "_session", "_runner", "_validation"]
] + [CLI]
TESTS = [
    "tests/python/test_installed_cost_8661.py",
    "tests/python/test_installed_cost_runner_8661.py",
]
ARMS = ["private_python", "private_native", "installed_python", "installed_native"]
GATES = [
    ("8653_v743_contract_methods", "exp8653-contract-methods", "deployment_inputs_ready_score"),
    (
        "8660_v743_installed_compatibility",
        "exp8660-installed-compatibility",
        "installed_measurement_ready_score",
    ),
]


def gate(path: Path, upstream: str, field: str) -> tuple[list[Json], Json]:
    """A missing field cannot masquerade as an observed numerical zero."""
    ref = reference(path) if path.is_file() else dict(path=str(path), sha256=None)
    value = load(ref) if ref["sha256"] else {}
    observed = value.get(field)
    state = (
        "missing_file" if not ref["sha256"] else "observed" if field in value else "missing_field"
    )
    rows = (
        []
        if type(observed) in {int, float} and observed == 1
        else [
            dict(
                upstream=upstream,
                check="upstream_readiness",
                **ref,
                field=field,
                operator="==",
                expected=1,
                observed=observed,
                observed_state=state,
            )
        ]
    )
    return rows, ref


def admit(root: Path) -> tuple[Json, list[Json], list[Json]]:
    """Original seals authorize numerical work; upstream costs never supply timings."""
    refs: list[Json] = []
    failures: list[Json] = []
    values = []
    for name, upstream, field in GATES:
        path = root / "results" / ("experiment_" + name + ".json")
        rows, ref = gate(path, upstream, field)
        failures.extend(rows)
        if ref["sha256"]:
            refs.append(ref)
            values.append(load(ref))
    if failures:
        return {}, failures, refs
    for ref, value in zip(refs[:], values, strict=True):
        side = (
            root
            / "results/raw"
            / Path(ref["path"]).stem
            / "validators"
            / (ref["sha256"][7:] + ".json")
        )
        if (
            not value["required_checks_passed"]
            or value["flagged_adversarial"]
            or not read_bound_sidecar(Path(ref["path"]), side)["report"]["passed"]
        ):
            raise ValueError("upstream_publication")
        refs.append(reference(side))
    methods, qualified = values
    protocol = load(methods["methods_protocol_reference"])
    method = protocol["installed_entry_2x2"]
    inherited = json.loads(
        (ROOT / "openspec/change-proposals/v741-methods-protocol.json").read_bytes()
    )["installed_entry_2x2"]
    if method != inherited or method != methods["installed_entry_2x2"]:
        raise ValueError("registered_method_drift")
    bundle = original_inputs(ROOT)
    request_ref = method["original_request_reference"]
    if reference(Path(request_ref["path"])) != request_ref:
        raise ValueError("request_hash")
    requests = json.loads(Path(request_ref["path"]).read_bytes())
    if requests != bundle["requests"]:
        raise ValueError("original_request_identity")
    from carnot.reporting.resident_provenance_8593 import check_descriptor

    if (
        not check_descriptor(qualified["target_model"])
        or bundle["target_model"] != qualified["target_model"]
    ):
        raise ValueError("head_descriptor")
    head = load(qualified["target_model"]["checkpoint"])["heads"]["spline9"][0]
    if head != bundle["head"] or len(head["parameters"]) != 109:
        raise ValueError("loaded_head")
    wheel = qualified["wheel_reference"]
    load_refs = [
        *bundle["refs"],
        methods["methods_protocol_reference"],
        method["original_request_reference"],
        wheel,
        reference(ROOT / "openspec/change-proposals/v741-methods-protocol.json"),
    ]
    with zipfile.ZipFile(wheel["path"]) as archive:
        for name in [
            "carnot/__init__.py",
            "carnot/inference/resident_protocol.py",
            "carnot/inference/resident_energy.py",
            "carnot/verify/source_alignment.py",
        ]:
            if archive.read(name) != (ROOT / "python" / name).read_bytes():
                raise ValueError("wheel_entry_drift")
            load_refs.append(reference(ROOT / "python" / name))
    load_refs.append(reference(ROOT / "scripts/experiments/resident_worker_8608.py"))
    load_refs.extend(
        reference(ROOT / name)
        for name in [
            "python/carnot/reporting/resident_provenance_8593.py",
            "python/carnot/verify/sparse_energy_7996.py",
            "python/carnot/verify/v726_decision_metrics.py",
        ]
    )
    for ref in load_refs:
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("frozen_operand_hash")
    return dict(bundle, wheel_reference=wheel, method=method), [], refs + load_refs


def roster() -> list[Json]:
    """Retain the registered rotation and alternating arithmetic order before timing."""
    result = []
    for length in [1, 4, 16]:
        for repeat in range(10):
            arms = ARMS[2:] + ARMS[:2] if repeat % 4 >= 2 else ARMS[:]
            if repeat % 2:
                arms = [arms[1], arms[0], arms[3], arms[2]]
            result.extend(
                dict(
                    unit_id=f"L{length}-B1-R{repeat}-{arm}",
                    length=length,
                    batch=1,
                    repeat=repeat,
                    arm=arm,
                )
                for arm in arms
            )
    return result


def interval(denominator: list[float], numerator: list[float]) -> list[float]:
    """Paired repeat indices retain dependence among all requests in one session."""
    indices = np.random.default_rng(7418635).integers(0, 10, size=(10000, 10))
    ratios = np.asarray(denominator)[indices].sum(axis=1) / np.asarray(numerator)[indices].sum(
        axis=1
    )
    return [float(x) for x in np.quantile(ratios, [0.025, 0.975])]


def scores(
    complete: bool, packaging: list[float], p95: float, arithmetic: list[float]
) -> tuple[int, int, int]:
    """Readiness, packaging noninferiority and tenfold arithmetic speed are distinct."""
    return (
        int(complete),
        int(complete and packaging[0] >= 0.95 and p95 <= 1.05),
        int(complete and arithmetic[0] >= 10),
    )


def summarize(primitive: Json, bundle: Json) -> tuple[Json, list[Json], list[Json]]:
    """Rebuild counts, outputs and nonoverlapping clocks from original frame operands."""
    slot = primitive["slot"]
    config = load(primitive["config_reference"])
    if (
        config["head"] != bundle["head"]
        or primitive["exit_code"] != 0
        or primitive["shutdown"]["status"] != "shutdown"
    ):
        raise ValueError("session_configuration_or_exit")
    ready = primitive["ready"]
    expected_extension = config["extension"] if slot["arm"].endswith("native") else None
    if (
        ready["status"] != "ready"
        or ready["config_hash"] != primitive["config_reference"]["sha256"]
        or ready["config_loads"] != 1
        or ready["loaded_extension"] != expected_extension
    ):
        raise ValueError("loaded_head_or_extension")
    rebuilt = [
        dict(
            operation="process_setup_import_configuration",
            start_ns=primitive["start_ns"],
            end_ns=primitive["ready_ns"],
        )
    ]
    requests = []
    references = {r["unit_id"]: r for r in bundle["references"]}
    observations = primitive["observations"]
    if len(observations) != 64 * slot["length"]:
        raise ValueError("request_count")
    for index, observed in enumerate(observations):
        operand = bundle["requests"][index % 64]
        frame, answer = observed["request"], observed["response"]
        identifier = f"{index // 64}-{index % 64}"
        if (
            frame["row"] != operand
            or frame["request_id"] != identifier
            or answer["request_id"] != identifier
            or answer["status"] != "ok"
            or frame["config_hash"] != primitive["config_reference"]["sha256"]
            or answer["config_hash"] != frame["config_hash"]
        ):
            raise ValueError("frame_operands")
        actual, expected = answer["response"], references[operand["unit_id"]]
        if (
            set(actual) != set(expected)
            or actual["unit_id"] != operand["unit_id"]
            or actual["decision"] != expected["decision"]
            or type(actual["boundary_fallback"]) is not bool
            or actual["boundary_fallback"]
            != (
                slot["arm"].endswith("native")
                and min(abs(actual["probability"] - 0.2), abs(actual["probability"] - 0.8)) <= 1e-10
            )
            or not math.isfinite(actual["probability"])
            or abs(actual["probability"] - expected["probability"])
            > 1e-12 * (1 + abs(expected["probability"]))
        ):
            raise ValueError("typed_output_parity")
        spans = answer["spans"]
        rebuilt.extend(
            [
                dict(
                    operation="parent_residual",
                    start_ns=rebuilt[-1]["end_ns"],
                    end_ns=observed["start_ns"],
                ),
                dict(
                    operation="transport_framing",
                    start_ns=observed["start_ns"],
                    end_ns=spans[0]["start_ns"],
                ),
                *spans,
                dict(
                    operation="transport_framing",
                    start_ns=spans[-1]["end_ns"],
                    end_ns=observed["end_ns"],
                ),
            ]
        )
        requests.append(
            dict(
                unit_id=operand["unit_id"],
                cell_id=slot["unit_id"],
                panel=index // 64,
                arm=slot["arm"],
                status="completed",
                response=actual,
                exact_decision=True,
                start_ns=observed["start_ns"],
                end_ns=observed["end_ns"],
                round_trip_s=(observed["end_ns"] - observed["start_ns"]) / 1e9,
                latency_s=(
                    observed["end_ns"] - (observed["start_ns"] if index else primitive["start_ns"])
                )
                / 1e9,
            )
        )
    rebuilt.append(
        dict(operation="clean_shutdown", start_ns=rebuilt[-1]["end_ns"], end_ns=primitive["end_ns"])
    )
    if any(s["end_ns"] < s["start_ns"] for s in rebuilt) or any(
        a["end_ns"] != b["start_ns"] for a, b in zip(rebuilt, rebuilt[1:])
    ):
        raise ValueError("clock_partition")
    duration = (primitive["end_ns"] - primitive["start_ns"]) / 1e9
    if duration <= 0:
        raise ValueError("session_clock")
    summary = dict(
        slot,
        status="completed",
        completed_count=len(requests),
        duration_s=duration,
        throughput_answers_s=len(requests) / duration,
        p95_request_latency_s=float(np.quantile([r["latency_s"] for r in requests], 0.95)),
        threshold_fallbacks=sum(r["response"]["boundary_fallback"] for r in requests),
        exact_decisions=True,
        host_load_before=primitive["host_load_before"],
        host_load_after=primitive["host_load_after"],
    )
    operations = [dict(cell_id=slot["unit_id"], **s) for s in rebuilt]
    return summary, requests, operations


def reduce(work: Json) -> Json:
    """An incomplete pair remains missing rather than entering a reduced sample bootstrap."""
    expected = roster()
    if [
        {k: row[k] for k in slot} for row, slot in zip(work["rows"], expected, strict=False)
    ] != expected or len(work["rows"]) != 120:
        raise ValueError("cell_roster")
    sessions, requests, operations = [], [], []
    for row in work["rows"]:
        if row["status"] != "completed":
            continue
        primitive = load(row["primitive_reference"])
        if primitive["slot"] != {k: row[k] for k in expected[0]}:
            raise ValueError("slot_identity")
        session, per_request, spans = summarize(primitive, work["bundle"])
        sessions.append(session)
        requests.extend(per_request)
        operations.extend(
            dict(
                cell_id=row["unit_id"],
                operation=name,
                duration_s=sum(s["end_ns"] - s["start_ns"] for s in spans if s["operation"] == name)
                / 1e9,
                span_count=sum(s["operation"] == name for s in spans),
            )
            for name in sorted({s["operation"] for s in spans})
        )
    contrasts = []
    for length in [1, 4, 16]:
        for contrast, numerator, denominator in [
            ("packaging", "installed_native", "private_native"),
            ("arithmetic", "installed_native", "installed_python"),
        ]:
            left = sorted(
                [r for r in sessions if r["length"] == length and r["arm"] == numerator],
                key=lambda r: r["repeat"],
            )
            right = sorted(
                [r for r in sessions if r["length"] == length and r["arm"] == denominator],
                key=lambda r: r["repeat"],
            )
            if len(left) != 10 or len(right) != 10:
                continue
            ci = interval([r["duration_s"] for r in right], [r["duration_s"] for r in left])
            latency = float(
                np.quantile(
                    [
                        r["latency_s"]
                        for r in requests
                        if r["cell_id"] in {x["unit_id"] for x in left}
                    ],
                    0.95,
                )
                / np.quantile(
                    [
                        r["latency_s"]
                        for r in requests
                        if r["cell_id"] in {x["unit_id"] for x in right}
                    ],
                    0.95,
                )
            )
            contrasts.append(
                dict(
                    contrast=contrast,
                    length=length,
                    batch=1,
                    numerator=numerator,
                    denominator=denominator,
                    paired_repeats=10,
                    lower95=ci[0],
                    upper95=ci[1],
                    p95_latency_ratio=latency,
                    draws=10000,
                    seed=7418635,
                )
            )
    primary = next(
        (r for r in contrasts if r["length"] == 16 and r["contrast"] == "packaging"), None
    )
    arithmetic = next(
        (r for r in contrasts if r["length"] == 16 and r["contrast"] == "arithmetic"), None
    )
    ready, packaging, nfr = scores(
        len(sessions) == 120,
        [primary["lower95"]] if primary else [0.0],
        primary["p95_latency_ratio"] if primary else math.inf,
        [arithmetic["lower95"]] if arithmetic else [0.0],
    )
    return dict(
        rows=work["rows"],
        intended_count=120,
        completed_count=len(sessions),
        failed_count=sum(r["status"] == "failed" for r in work["rows"]),
        censored_count=sum(r["status"] == "censored" for r in work["rows"]),
        excluded_count=0,
        independent_count=64 if sessions else 0,
        per_request_rows=requests,
        per_session_rows=sessions,
        paired_throughput_intervals=contrasts,
        p95_latency_ratios=[
            {k: r[k] for k in ["contrast", "length", "p95_latency_ratio"]} for r in contrasts
        ],
        complete_operation_spans=operations,
        installed_cost_ready_score=ready,
        packaging_noninferiority_score=packaging,
        nfr01_score=nfr,
    )


def empty_work(raw: Path, run_date: str) -> Json:
    """Every intended cell exists before admission, so blocking preserves its denominator."""
    return dict(
        raw=str(raw),
        run_date=run_date,
        bundle={},
        gate_check_summary=[],
        rows=[dict(slot, status="censored", missing_reason="not_started") for slot in roster()],
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        validation_receipts=[],
        loaded_module_hashes=[],
        phase_spans=[],
        duration_s=0.0,
        preconditions_checked={},
        owned_error=None,
    )


def artifact(authority: Json) -> Json:
    """Only complete validated measurements earn readiness; a valid null keeps its data."""
    work = load(authority)
    value = reduce(work)
    required = {
        "owned_tests",
        "exact_node_union",
        "owned_coverage",
        "e2e018",
        "e2e021",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    }
    receipts = work["validation_receipts"]
    checks = required <= {r["name"] for r in receipts if r["passed"]} and all(
        r["passed"] for r in receipts
    )
    verdict = (
        "blocked"
        if work["gate_check_summary"]
        else "partial"
        if value["completed_count"] < 120
        else "null"
    )
    if (
        work["owned_error"]
        or any(not r["passed"] for r in receipts)
        or (verdict == "null" and not checks)
    ):
        verdict = "disqualified"
    if verdict in {"blocked", "partial", "disqualified"}:
        value.update(installed_cost_ready_score=0, packaging_noninferiority_score=0, nfr01_score=0)
    if value["packaging_noninferiority_score"] and value["nfr01_score"]:
        verdict = "circular_positive"
    bundle = work["bundle"]
    value.update(
        experiment_id=8661,
        task_id=TASK,
        milestone="2026.10.743",
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_v743_installed_complete_cost",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts={
            "language_model_calls": 0,
            "model_loads": 0,
            "numerical_head_requests": len(value["per_request_rows"]),
        },
        current_language_model_calls=0,
        historical_model_provenance=bundle.get("historical_model_provenance", []),
        target_model=bundle.get("target_model"),
        verifier_is_oracle=True,
        exposure_scope="original64 exposed development requests; original checkpoint; no independent generalization or hardware claim",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        sample_size_budget=dict(
            intended_cells=120,
            original_request_identities=64,
            independent_questions_max=64,
            repeats=10,
            lengths=[1, 4, 16],
            batch=1,
            intended_request_evaluations=53760,
        ),
        required_checks_passed=checks,
        flagged_adversarial=False,
        adversarial_findings=[],
        acceptance_gates=[
            dict(
                name="qualified_complete_measurement",
                passed=bool(value["installed_cost_ready_score"]),
            ),
            dict(
                name="packaging_noninferiority",
                passed=bool(value["packaging_noninferiority_score"]),
            ),
            dict(name="NFR-01", passed=bool(value["nfr01_score"])),
        ],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        authority_reference=authority,
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7418635,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=work["raw_shard_hashes"],
        cited_upstream_artifacts=work["source_artifact_hashes"],
        loaded_module_hashes=work["loaded_module_hashes"],
        warmup="No discarded timed sessions; fresh process and full startup in every cell. Untimed identity probe is retained.",
        preserved_historical_nulls=[
            "Exp8622 V740 control only",
            "Exp8635 blocked",
            "Exp8647 blocked",
            "Exp8646 disqualified; guarded teardown",
        ],
        preserved_quarantines=bundle.get("preserved_quarantines", []),
        owned_error=work["owned_error"],
        methodology="Fixed original64 complete text requests; private/installed Python/native L1/4/16 B1 ten registered paired repeats. Full launch-to-clean-exit throughput, per-request round trips and first-request startup latency. Disjoint raw operation clocks; 10000 paired-index bootstrap draws seed7418635. Exposed checkpoint; no hardware or generalization claim.",
    )
    value["branch_gate_values"] = {
        key: value[key]
        for key in ["installed_cost_ready_score", "packaging_noninferiority_score", "nfr01_score"]
    }
    value["field_principles"] = {
        key: "Bind actual immutable invocation evidence and retain missing status; engineering readiness and exposed timing grant no generalization benefit."
        for key in value
    }
    value["field_principles"].update(
        installed_cost_ready_score="All120 equal-work cells and owned checks qualify measurement.",
        packaging_noninferiority_score="L16 installed_native/private_native lower95>=0.95, p95<=1.05 and exact decisions.",
        nfr01_score="Only L16 installed_native/installed_python lower95>=10 satisfies NFR-01.",
        independent_count="At most64 original question identities; panels, arms and repeats are dependent.",
        complete_operation_spans="Aggregate disjoint raw operation spans by cell; raw frames retain every endpoint.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path, authority: Json) -> None:
    """An out-of-band authority hash rejects self-consistent changes to claims or operands."""
    work = load(authority)
    for ref in (
        work["source_artifact_hashes"] + work["code_config_hashes"] + work["raw_shard_hashes"]
    ):
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("replay_operand_hash")
    if json.loads(path.read_bytes()) != artifact(authority):
        raise ValueError("immutable_reduction_drift")
