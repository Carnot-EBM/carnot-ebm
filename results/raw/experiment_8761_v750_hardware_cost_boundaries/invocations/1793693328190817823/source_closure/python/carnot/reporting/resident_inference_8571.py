"""REQ-REPORT-8571: original authority and complete-cost null constrain this prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import complete_native_cost_8559 as cost
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8571_v736_resident_inference_prototype"
TASK = "exp8571-resident-inference-prototype"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_resident_inference_8571.py"
OWNED = [
    f"python/carnot/reporting/resident_{name}_8571.py"
    for name in ["worker", "inference", "prototype", "runner", "validation"]
] + [CLI]
PINS = {
    "results/experiment_8558_v735_nine_input_native_qualification.json": cost.PIN,
    "results/experiment_8559_v735_complete_native_cost.json": "sha256:d08c9868dc613dd1b8f7ee7e53a2e6e5497d16a2526dcf7e85e13e35eff7862c",
}
GateFailure, require = cost.GateFailure, cost.require


def reference(path: Path) -> Json:
    """Hash exact file bytes so no later reader trusts a mutable label alone."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read(ref: Json) -> Any:
    """Missing and changed evidence fail with both gate operands retained."""
    path = Path(ref["path"])
    cost.require(path, "input_hash", ref["sha256"], sha256_file(path) if path.is_file() else None)
    return json.loads(path.read_bytes())


def admit(root: Path) -> Json:
    """Authenticate current primaries, their publication, and the unchanged numeric operands."""
    originals = [read(dict(path=str(root / p), sha256=h)) for p, h in PINS.items()]
    parity, timing = originals
    refs = [dict(path=str(root / p), sha256=h) for p, h in PINS.items()]
    for value, ref in zip(originals, list(refs), strict=True):
        path = Path(ref["path"])
        for field, expected in dict(required_checks_passed=True, flagged_adversarial=False).items():
            cost.require(path, field, expected, value.get(field))
        terminal = Path(value["terminal_validation_sidecar_path"])
        sidecar = Path(json.loads(terminal.read_bytes())["publication"]["sidecar_path"])
        cost.require(
            path, "bound_publication", True, read_bound_sidecar(path, sidecar)["report"]["passed"]
        )
        refs.extend([reference(terminal), reference(sidecar)])
    head, tape = read(parity["head_reference"]), read(parity["request_tape_reference"])
    refs.extend(
        [parity["head_reference"], parity["request_tape_reference"], parity["extension_reference"]]
    )
    for ref in parity["extension_reference"]["sources"]:
        cost.require(
            Path(ref["path"]), "native_source", ref["sha256"], sha256_file(Path(ref["path"]))
        )
        refs.append(ref)
    cost.require(Path(refs[0]["path"]), "parity_ready", 1, parity["nine_input_native_ready_score"])
    cost.require(
        Path(parity["extension_reference"]["path"]),
        "extension_hash",
        parity["extension_reference"]["sha256"],
        sha256_file(Path(parity["extension_reference"]["path"])),
    )
    work = read(timing["work_reference"])
    reduced = cost.reduce(work)
    cost.require(
        Path(refs[1]["path"]), "original_cost_rows", timing["startup_cost"], reduced["startup_cost"]
    )
    cost.require(
        Path(refs[1]["path"]),
        "original_operation_clocks",
        timing["operation_shares"],
        reduced["operation_shares"],
    )
    refs.append(timing["work_reference"])
    protocol_path = ROOT / "openspec/change-proposals/v736-methods-protocol.json"
    cost.require(
        protocol_path,
        "frozen_protocol_hash",
        "sha256:88bbf528a7206e67bbee730ef1fb71d7d7ccb27c7ef3b442430c4ff51026f434",
        sha256_file(protocol_path),
    )
    protocol = json.loads(protocol_path.read_bytes())
    for key in [
        "v736_activation_reference",
        "observed_design_reference",
        "inherited_v735_protocol_reference",
    ]:
        read_ref = protocol[key]
        cost.require(
            Path(read_ref["path"]), key, read_ref["sha256"], sha256_file(Path(read_ref["path"]))
        )
        refs.append(read_ref)
    refs.append(reference(protocol_path))
    for row in work["rows"]:
        for key in ["reference", "primitive_reference"]:
            if key in row:
                refs.append(row[key])
    requests = [r for r in tape["rows"] if r["unit_id"] in tape["timed_ids"]]
    cost.require(
        protocol_path, "stable_request_ids", tape["timed_ids"], [r["unit_id"] for r in requests]
    )
    return dict(
        head=head,
        requests=requests,
        extension=parity["extension_reference"],
        refs=refs,
        historical_model_provenance=parity["historical_model_provenance"],
        historical_cost_null=dict(
            ratio=timing["complete_throughput_ratio"],
            ci95=timing["throughput_ratio_ci95"],
            verdict_class=timing["verdict_class"],
            startup_rows=timing["startup_cost"],
            operation_rows=timing["operation_shares"],
            new_timing_comparison=False,
        ),
        original_authority=protocol["v736_activation_reference"],
    )


def checksum(value: Json) -> str:
    """A checksum binds the envelope; cold replay separately checks its meaning."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def prototype(bundle: Json, raw: Path) -> Json:
    """An explicit adapter keeps protocol code separate from historical admission."""
    from carnot.reporting.resident_prototype_8571 import execute

    return execute(bundle, raw)


def replay_primitive(primitive: Json, bundle: Json) -> bool:
    """Regenerate responses, never timings, using the original head and public text."""
    from carnot.reporting.resident_prototype_8571 import replay

    return bool(replay(primitive, bundle))


REQUIRED = {
    "exact_disjoint_node_union",
    "owned_python_coverage_100",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "cold",
    "missing",
    "owned_failure",
    "rehashed_tamper",
    "adversarial",
    "strict_rows",
}


def literature() -> list[Json]:
    """Versioned primary bytes justify method choices, not local speed or learning claims."""
    path = ROOT / "results/raw" / NAME / "literature/sources.json"
    sources = json.loads(path.read_bytes())
    pins = {
        "2602.02056v4": "sha256:ad9d1e80c8a599d6c7994219eacbe654361cb4f0b6d9844c17033bce5e891e1f",
        "2604.22034v1": "sha256:cb020caa0b1637a689b21ec34d2190a2dbbac4910793ea7f3d4899e70f8471c9",
        "2606.19808v1": "sha256:236dfdc4e5053c81e983610df1d5f71b2800a8a0f048efe4294f5f481400aa8f",
    }
    cost.require(path, "primary_method_order", list(pins), [s["paper"] for s in sources])
    mappings = [
        (
            "Local B-spline support; batch-one updates; fixed-point and resource evaluation.",
            "nine_input_8533.probabilities and sparse_energy_7996.parameters preserve109 coefficients; no updates here.",
            "No FPGA, quantization, regret or online-learning result transfers to this CPU prototype.",
        ),
        (
            "Bernstein-Bezier basis evaluation, grid adaptation and factorized layers.",
            "sparse_energy_7996.basis and qualified native residual remain the fixed reference; replacement deferred.",
            "No new basis, grid adaptation, parameter reduction or published speed ratio is implemented.",
        ),
        (
            "Total realized serving cost; helpful fixes and harmful flips; matched workload comparisons.",
            "complete_native_worker_8559.batch retains extraction through serialization; prototype compares every typed decision.",
            "Count harmful decision changes against shipped responses; full serving-cost qualification and policy training are deferred.",
        ),
    ]
    rows = []
    for source, (method, mapping, limitation) in zip(sources, mappings, strict=True):
        cost.require(
            Path(source["path"]), "retrieved_source_pin", pins[source["paper"]], source["sha256"]
        )
        cost.require(
            Path(source["path"]),
            "literature_hash",
            source["sha256"],
            sha256_file(Path(source["path"])),
        )
        rows.append(dict(source, method=method, code_mapping=mapping, limitation=limitation))
    return rows


def artifact(work: Json, receipts: list[Json], output: Path) -> Json:
    """Readiness qualifies a runnable lifecycle only; no speed or scientific benefit is inferred."""
    bundle, primitive = work["bundle"], work["primitive"]
    checks = {r["name"]: r["passed"] for r in receipts}
    required = REQUIRED <= checks.keys() and all(r["passed"] for r in receipts)
    rows = primitive["prototype_request_rows"] if primitive else []
    clean = bool(primitive) and len(rows) == 8 and all(r["passed"] for r in rows)
    ready = clean and required and not work["gate_check_summary"]
    verdict = (
        "disqualified"
        if any(not r["passed"] for r in receipts)
        else "blocked"
        if work["gate_check_summary"]
        else "null"
        if ready
        else "disqualified"
    )
    raw = Path(work["raw"])
    value: Json = dict(
        experiment_id=8571,
        task_id=TASK,
        milestone="2026.10.736",
        run_date="20261016",
        honest_verdict=f"complete_{verdict}_v736_resident_inference_prototype",
        verdict_class=verdict,
        gate_check_summary=work["gate_check_summary"]
        + [
            dict(
                check=r["name"],
                upstream=TASK,
                path=r.get("stdout_path"),
                sha256=r.get("stdout_sha256"),
                field="passed",
                operator="==",
                expected=True,
                observed=False,
            )
            for r in receipts
            if not r["passed"]
        ],
        inference_substrate="CPU transformation of frozen authentic outputs, no LLM",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_llm_invocations=0,
        historical_model_provenance=bundle["historical_model_provenance"] if bundle else [],
        rows=rows,
        intended_count=8,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            private_requests=8, sessions=4, independent_workloads=0, comparative_timing_trials=0
        ),
        verifier_is_oracle=True,
        exposure_scope="original exposed test-role public answer records; private lifecycle controls",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        prototype_ready_score=int(ready),
        required_checks_passed=required,
        flagged_adversarial=bool(work.get("adversarial_findings")) or verdict == "disqualified",
        acceptance_gates=dict(
            bounded_cli=clean,
            required_validation=required,
            complete_service_speedup_measured=False,
            harmful_interventions_allowed=0,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7368571,
        source_artifact_hashes=bundle["refs"] if bundle else [],
        code_config_hashes=[reference(ROOT / p) for p in [*OWNED, TEST]],
        raw_shard_hashes=[
            reference(path)
            for path in [
                raw / "prototype" / "prototype.json",
                raw / "collection.json",
                raw / "node_union.json",
                raw / "coverage.json",
            ]
            if path.is_file()
        ],
        cited_upstream_artifacts=list(PINS),
        literature_method_rows=literature(),
        immutable_config_reference=primitive["immutable_config_reference"] if primitive else None,
        prototype_request_rows=rows,
        lifecycle_receipts=primitive["lifecycle_receipts"] if primitive else [],
        historical_cost_null=bundle["historical_cost_null"] if bundle else None,
        worker_protocol=dict(
            version=1,
            transport="local pipes",
            framing="uint32 big-endian byte length then JSON UTF-8",
            max_frame_bytes=65536,
            max_requests_per_session=64,
            queue_capacity=1,
            request_deadline_s=15,
            idle_deadline_s=15,
            startup_deadline_s=30,
            shutdown_deadline_s=10,
            cache=False,
            configuration_loads_per_session=1,
            changed_variable="process lifetime only",
            production_default=False,
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8572, 8579)
        ],
        harmful_intervention_criterion="Any typed decision change from shipped complete evaluator fails this prototype; this is numerical parity, not correctness labeling.",
        harmful_intervention_count=sum(r["kind"] == "valid" and not r["passed"] for r in rows),
        full_serving_cost_criterion="Future qualification includes launch, transport, parsing, extraction, arithmetic, normalization, serialization and shutdown; primary16 panels/batch1 with paired-repeat CI and p95<=1.05.",
        work_reference=reference(raw / "work.json"),
        publication_output=str(output.absolute()),
    )
    value["field_principles"] = {
        k: "Bind this invocation to immutable original operands; readiness grants comprehensive qualification only, never speed or learning benefit."
        for k in value
    }
    value["field_principles"].update(
        completed_count="Eight protocol attempts include six valid evaluations and two intentional rejections; repeated inputs add no independence.",
        historical_cost_null="Original complete launch-to-response clocks are imported and decomposed without retiming; CI95 includes1.",
        prototype_ready_score="One only when bounded real CLI, unchanged decisions, restart and every required owned check pass.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct exact summary bytes from authenticated original inputs and real child primitives."""
    try:
        value = json.loads(path.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            return False
        work = read(value["work_reference"])
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            cost.require(
                Path(ref["path"]), "replay_hash", ref["sha256"], sha256_file(Path(ref["path"]))
            )
        if work["bundle"] and (
            work["bundle"] != admit(ROOT) or not replay_primitive(work["primitive"], work["bundle"])
        ):
            return False
        return bool(
            artifact(work, value["validation_receipts"], Path(value["publication_output"])) == value
        )
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False
