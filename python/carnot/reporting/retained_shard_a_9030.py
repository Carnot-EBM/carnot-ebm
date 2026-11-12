"""REQ-REPORT-9030-CUSTODY: source custody cannot undo published test outcomes."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

from carnot.reporting import graph_corpus_a_8990 as shard
from carnot.inference import graph_corpus_runtime_8961 as capture
from carnot.reporting import graph_source_8931 as source
from carnot.reporting import invocation_provenance_9000 as original
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)

Json = dict[str, Any]
ROOT = original.ROOT
NAME, TASK = "experiment_9030_v771_retained_shard_a", "exp9030-retained-shard-a"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/retained_shard_a_9030.py",
    "python/carnot/reporting/retained_shard_runner_9030.py",
    CLI,
]
TEST = "tests/python/test_retained_shard_a_9030.py"
PINS = {
    "results/experiment_9028_v771_historical_provenance_qualification.json": (
        "sha256:a2cb53fd67a42e3bb9614849c42f27fc1d694a65bdc5d508ac5943a77e774067",
        "provenance_ready_score",
    ),
    "results/experiment_9027_v771_branch_contract.json": (
        "sha256:3c233de7321f62d1286e8e94bd045da059bc1d65a33f503b637dbd904d084ff7",
        "frozen_graph_authority_ready_score",
    ),
}
reference, retain, checksum, gate = (
    original.reference,
    original.retain,
    original.checksum,
    original.gate,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report real work counts without extending any deadlines."""
    print(f"[exp9030] phase={phase} completed={completed} pending={pending}", flush=True)


def audit(root: Path, raw: Path) -> Json:
    """Reopen immutable observations; a headline ready bit never supplies a response."""
    progress("before_authentication", 0, 512)
    raw.mkdir(parents=True, exist_ok=True)
    evidence: Json = dict(
        checks=[],
        references=[],
        rows=[],
        role_counts={},
        raw_receipt_audit={},
        nonexposure_audit={},
        original_disqualified_primary={},
        downstream_evaluation_attempts=[],
    )
    checks, refs = evidence["checks"], evidence["references"]
    for name, (pin, field) in PINS.items():
        path = root / name
        checks.append(
            gate(
                path,
                "sha256",
                pin,
                reference(path)["sha256"] or "missing_file",
                "current_authority_bytes",
            )
        )
        if path.is_file():
            refs.append(retain(path, raw))
        if checks[-1]["passed"]:
            value = json.loads(path.read_bytes())
            checks.extend(
                gate(path, k, expected, value.get(k, "missing_field"), "current_authority_gate")
                for k, expected in [
                    (field, 1),
                    ("required_checks_passed", True),
                    ("flagged_adversarial", False),
                ]
            )
    if any(not r["passed"] for r in checks):
        return evidence
    authenticated = original.authenticate(root, raw)
    checks.extend(authenticated["checks"])
    refs.extend(authenticated["references"])
    if authenticated["failures"]:
        return evidence
    primary_path = root / original.INPUTS[0][0]
    primary = json.loads(primary_path.read_bytes())
    work = json.loads(Path(primary["raw_shard_hashes"][0]["path"]).read_bytes())
    units = shard.ordered_units(work["authentication"]["protocol"])
    runtime = Path(work["raw"]) / "runtime"
    checks.append(
        gate(
            primary_path,
            "units",
            canonical_hash(units),
            canonical_hash(work["units"]),
            "committed_roster",
        )
    )
    rebuilt = []
    for index, (unit, prior) in enumerate(zip(units, work["runtime"]["rows"], strict=True)):
        folder = capture.folder(runtime / "collection", index, unit)
        envelope = json.loads((folder / "response.json").read_bytes())
        row = source.label_response(unit, envelope["raw_response"])
        row.update(
            duration_s=envelope["duration_s"], actual_token_usage=envelope["raw_response"]["usage"]
        )
        fields = [
            "unit_id",
            "answer",
            "parser_category",
            "value",
            "numerator",
            "features",
            "status",
        ]
        checks.append(
            gate(
                folder / "response.json",
                "parser_and_label",
                {k: prior[k] for k in fields},
                {k: row[k] for k in fields},
                "independent_response_reduction",
            )
        )
        rebuilt.append(row)
        if index % 64 == 0:
            progress("response_reduction", index + 1, 511 - index)
    reduced = shard.reduce(units, rebuilt)
    checks.append(
        gate(primary_path, "labels_verified", True, reduced["labels_verified"], "independent_bfs")
    )
    checkpoint = runtime / "collection/checkpoint.json"
    saved = json.loads(checkpoint.read_bytes())
    checks.append(
        gate(
            checkpoint,
            "rows",
            [
                {k: v for k, v in r.items() if k != "actual_token_usage"}
                for r in work["runtime"]["rows"]
            ],
            saved["rows"],
            "checkpoint_ledger",
        )
    )
    model = work["runtime"]["model_receipt"]
    for field, expected in [
        ("gguf_sha256", "sha256:7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"),
        ("tokenizer", "embedded_GGUF"),
        ("offload_layers", [66, 66]),
    ]:
        checks.append(
            gate(
                runtime / "runtime.json",
                field,
                expected,
                model.get(field, "missing_field"),
                "original_model_receipt",
            )
        )
    published = [
        r["unit_id"]
        for r in primary["response_rows"]
        if r["split"] == "test" and r["value"] is not None
    ]
    checks.append(
        gate(
            primary_path, "response_rows[split=test].value", 0, len(published), "nonexposure_audit"
        )
    )
    for number in [*range(8991, 8998), *range(9002, 9011), *range(9016, 9025)]:
        for path in sorted((root / "results").glob(f"experiment_{number}_*.json")):
            value = json.loads(path.read_bytes())
            ref = retain(path, raw)
            refs.append(ref)
            evidence["downstream_evaluation_attempts"].append(
                dict(
                    experiment_id=number,
                    reference=ref,
                    honest_verdict=value.get("honest_verdict"),
                    verdict_class=value.get("verdict_class"),
                    model_invocation_counts=value.get("model_invocation_counts", {}),
                )
            )
    rows = [
        dict(r, group_id=r["cluster_id"], exclusion_reason=r["missing_reason"])
        for r in reduced["rows"]
    ]
    evidence.update(
        repository_health=work["repository_health"],
        rows=rows,
        role_counts=reduced["role_counts"],
        label_rows=reduced["label_sidecar"],
        role_manifest_hash=shard.ROLE_PIN,
        original_generation_duration_s=sum(
            r["duration_s"] for r in work["runtime"]["phase_spans"] if r["phase"] == "generation"
        ),
        original_model_receipt_reference=reference(runtime / "runtime.json"),
        original_disqualified_primary=dict(
            reference(primary_path),
            honest_verdict=primary["honest_verdict"],
            verdict_class=primary["verdict_class"],
            flags=authenticated["original_flag_rows"],
            flag_scope_explanation="Original nested invocation/authentication serialization contradicted its current live scope. This audit cites those observations as upstream, retains both flags, and performs zero model calls.",
        ),
        raw_receipt_audit=dict(
            passed=all(r["passed"] for r in checks if r["check"] != "nonexposure_audit"),
            historical_counts=authenticated["historical_counts"],
            checkpoint_reference=reference(checkpoint),
            original_receipt_reference=reference(runtime / "runtime.json"),
        ),
        nonexposure_audit=dict(
            passed=False,
            published_test_outcomes=len(published),
            exposed_unit_ids=published,
            detector_fitting_proven=False,
            closure="No complete post-collection access ledger proves non-exposure. All64 test outcomes were published by the original primary; missing downstream primaries do not prove absence of evaluation.",
        ),
    )
    atomic_json(raw / "response_rows.json", dict(rows=rows))
    atomic_json(raw / "labels.json", dict(rows=reduced["label_sidecar"]))
    refs.extend(reference(raw / p) for p in ["response_rows.json", "labels.json"])
    progress("after_audit", 512, 0)
    return evidence


def build(
    evidence: Json,
    receipts: list[Json],
    coverage: Json,
    scope: Json,
    duration: float,
    *,
    accepted: bool,
) -> Json:
    """Current acceptance qualifies the audit, while independent gates control admission."""
    failures = [r for r in evidence["checks"] if not r["passed"]]
    cls = "disqualified" if not accepted else "blocked" if failures else "circular_positive"
    rows = evidence["rows"]
    value: Json = dict(
        experiment_id=9030,
        experiment=9030,
        task_id=TASK,
        title="Retained shard A custody audit",
        milestone="2026.11.771",
        run_date="20261112",
        honest_verdict="complete_"
        + ("blocked_" + failures[0]["check"] if cls == "blocked" else cls + "_retained_shard_a"),
        verdict_class=cls,
        status="blocked" if cls == "blocked" else "error" if cls == "disqualified" else "success",
        shard_ready_score=int(cls == "circular_positive"),
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="Historical source custody only. BFS labels are circular oracle evidence; no detector benefit or independent generalization.",
        exposure_scope="Original calibration/test/online responses; final-test unit outcomes already published; non-exposure not certified.",
        required_checks_passed=accepted,
        flagged_adversarial=not accepted,
        validation_receipts=receipts,
        owned_statement_coverage=coverage,
        validation_scope_manifest=scope,
        repository_health=dict(
            passed=False,
            original_failed_receipts=evidence.get("repository_health", {}),
            scope="Historical repository health remains failed; current checks cover this audit and frozen direct consumers only.",
            reference=evidence.get("original_disqualified_primary", {}),
        ),
        repository_health_passed=False,
        gate_check_summary=evidence["checks"],
        preconditions_checked=evidence.get("environment", {}),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_mode="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        model_invoked=False,
        numerical_training_calls=0,
        minimum_duration_s=0,
        rows=rows,
        response_rows=rows,
        first_response_rows=rows,
        role_counts=evidence["role_counts"],
        summary_recomputation=dict(
            parser_categories=dict(Counter(r["parser_category"] for r in rows)),
            error_count=sum(r["numerator"] for r in rows),
            denominator=len(rows),
            method="Independent reduction of retained per-unit parser and BFS rows.",
        ),
        intended_count=512,
        independent_count=len(rows),
        completed_count=len(rows),
        excluded_count=0,
        censored_count=512 - len(rows),
        failed_count=0,
        duration_s=duration,
        random_seed=771030,
        phase_spans=evidence.get("phase_spans", []),
        source_artifact_hashes=evidence["references"],
        raw_shard_hashes=evidence.get("work_references", []),
        code_config_hashes=scope.get("code_test_hashes", []),
        raw_receipt_audit=evidence["raw_receipt_audit"],
        nonexposure_audit=evidence["nonexposure_audit"],
        original_disqualified_primary=evidence["original_disqualified_primary"],
        role_manifest_hash=evidence.get("role_manifest_hash"),
        original_model_receipt_reference=evidence.get("original_model_receipt_reference"),
        original_generation_duration_s=evidence.get("original_generation_duration_s"),
        downstream_evaluation_attempts=evidence["downstream_evaluation_attempts"],
        acceptance_gates=dict(
            current_normal_exit_checks=True,
            zero_exclusion_owned_coverage_percent=100,
            provenance_ready_score=1,
            frozen_graph_authority_ready_score=1,
            unique_A_units=512,
            calibration=64,
            final_test=64,
            online=384,
            unexposed_test_outcomes=0,
        ),
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(n).name.split("_")[1]),
                sha256=p,
                fields_imported=[
                    "historical observations and immutable receipts"
                    if "8990" in n
                    else "frozen graph/role/state authority"
                    if "8987" in n
                    else f
                ],
            )
            for n, p, f in [
                *((n, p, f) for n, (p, f) in PINS.items()),
                *((n, p, "original_generation") for n, p in original.INPUTS),
            ]
        ],
        methodology_note="No generation, fitting, relabeling, replacement or score selection. Reopen all original response bytes and ledger receipts; reproduce parser and independent shortest distances. Published test outcomes prevent non-exposure certification.",
        reproducibility_checksum="",
    )
    value["field_principles"] = {
        k: "Retained primitive bytes and actual normal exits determine current custody; historical calls and failures remain cited upstream."
        for k in value
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
