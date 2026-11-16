"""REQ-REPORT-9122-V777: account for unavailable complete services honestly.

A missing scientific producer cannot be repaired by loading Qwen again. This
inventory retains exact operands and original TEST units, with no service clock
inferred from the time spent checking files. Eligible science requires a real
measurement worker before this inventory may be published.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
from typing import Any

from carnot.inference.sota_models import cached_current_model
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.hardware_boundary_9079 import CONDITIONS
from carnot.reporting.service_cost_9093 import cost_row, gate, ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, SEED = "experiment_9122_v777_complete_cost", "exp9122-complete-cost", 776108
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/complete_cost_9122.py",
    "python/carnot/reporting/complete_cost_runner_9122.py",
    CLI,
]
TESTS = ["tests/python/test_complete_cost_9122.py"]
MODEL_SPECS = [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
INPUTS = {
    9112: (
        "results/experiment_9112_v777_masked_runtime.json",
        ["embeddings_ready_score", "logits_ready_score"],
    ),
    9117: ("results/experiment_9117_v777_static_heads.json", ["static_predictions_ready_score"]),
    9118: ("results/experiment_9118_v777_online_streams.json", ["online_predictions_ready_score"]),
    9121: ("results/experiment_9121_v777_energy_audit.json", ["energy_audit_ready_score"]),
}
AUTHORITY = (
    "results/raw/experiment_9110_v777_contract/invocations/1794778518146924127/"
    "authority/results/raw/experiment_9109_v776_capstone/"
    "invocations/1794766780181981018/retained/"
)
COHORT_INPUTS = {
    "manifest": (
        AUTHORITY
        + "cb6b456ffe0507872a04264166a60c07a772011e04f76de6d99552fa3565e532/source_manifest.json",
        "sha256:cb6b456ffe0507872a04264166a60c07a772011e04f76de6d99552fa3565e532",
    ),
    "roles": (
        AUTHORITY
        + "bb868af1d6f173c5b98155498e5daadd286e2641ef4aa3c20b7a0671b1b95117/v726-methods-protocol.json",
        "sha256:bb868af1d6f173c5b98155498e5daadd286e2641ef4aa3c20b7a0671b1b95117",
    ),
}
_arm_rng = random.Random(SEED)
PROTOCOL: Json = dict(
    process_arm_order=[_arm_rng.sample(["embedding", "energy"], k=2) for _ in range(3)],
    qids=20,
    answers=60,
    process_repetitions=3,
    arm_order_seed=SEED,
    parity_atol=1e-5,
    parity_rtol=1e-4,
    typed_decisions="exact",
    generated_tokens=0,
    minimum_actual_substrate_s=2,
    measurement_deadline_s=3300,
    validation_reserve_s=1200,
    total_cap_s=4800,
    embedding="raw text/tokenize/masked features/transforms/projected-linear/spline/full-linear/serialize",
    energy="raw text/tokenize/all teacher-forced statistics/frozen probability and energy heads/serialize",
    cache_conditions=["cold_load", "warm_uncached", "question_cache_hit", "cached_head_only"],
    adaptation=[
        "proposal",
        "full_CAL_guard",
        "rollback",
        "final_feedback_drain",
        "fsync",
        "recovery",
    ],
    model_admission=[
        "cached_current_model(gpu_index=0)",
        "GPU0 lease",
        "free VRAM",
        "original GGUF/tokenizer/build hashes",
        "actual CUDA offload",
    ],
)


def progress(phase: str) -> None:
    """Flush boundaries so even a refusal remains visible to its supervisor."""
    print(f"[exp9122] phase={phase}", flush=True)


def checksum(value: Json) -> str:
    """Bind every result field while avoiding a self-referential checksum."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def read_reference(reference: Json) -> Json:
    """Rehash retained bytes before reading; a checkout replacement is never used."""
    path = Path(reference["path"])
    if ref(path)["sha256"] != reference["sha256"]:
        raise ValueError("retained_hash_drift")
    return dict(json.loads(path.read_bytes()))


def retain(path: Path, raw: Path, references: list[Json]) -> Json:
    """Freeze exact observed bytes once so later checkout changes cannot affect replay."""
    data = path.read_bytes()
    digest = canonical_hash(data.hex())
    target = raw / "objects" / digest.split(":")[1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.chmod(0o400)
    reference = dict(path=str(target), sha256=sha256_file(target), source_path=str(path))
    references.append(reference)
    return read_reference(reference)


def collect(root: Path, raw: Path) -> Json:
    """Check both scientific branches before any load, retaining only authenticated data."""
    progress("before_preconditions")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_json(raw / "protocol.json", PROTOCOL)
    checks: list[Json] = []
    references: list[Json] = []
    values: dict[int, Json] = {}
    available: dict[tuple[int, str], bool] = {}
    for identity, (label, fields) in INPUTS.items():
        path = root / label
        digest = ref(path)["sha256"]
        try:
            value = retain(path, raw, references) if digest else {}
        except (ValueError, TypeError):
            value = {}
            checks.append(
                gate(
                    str(path),
                    "input_format",
                    "JSON object",
                    "invalid_json",
                    "input_format",
                    identity,
                    digest,
                )
            )
        identity_ok = type(value.get("experiment_id")) is int and value["experiment_id"] == identity
        if digest:
            checks.append(
                gate(
                    str(path),
                    "experiment_id",
                    identity,
                    value.get("experiment_id", "missing_field"),
                    "identity",
                    identity,
                    digest,
                )
            )
        bound = True
        if digest:
            check = gate(
                str(path),
                "required_checks_passed",
                True,
                value.get("required_checks_passed", "missing_field"),
                "upstream_validation",
                identity,
                digest,
            )
            checks.append(check)
            bound = check["passed"]
        for binding in value.get("code_config_hashes", []) + value.get("raw_shard_hashes", []):
            original = Path(binding["path"])
            original = original if original.is_absolute() else root / original
            observed = ref(original)["sha256"] or "missing_file"
            check = gate(
                str(original),
                "sha256",
                binding["sha256"],
                observed,
                "producer_binding",
                identity,
                ref(original)["sha256"],
            )
            checks.append(check)
            bound = bound and check["passed"]
        for field in fields:
            check = gate(
                str(path),
                field,
                1,
                value.get(field, "missing_field") if digest else "missing_file",
                "upstream_readiness",
                identity,
                digest,
            )
            checks.append(check)
            available[identity, field] = check["passed"] and identity_ok and bound
        values[identity] = value
        progress(f"after_source_{identity}")
    cohort: dict[str, Json] = {}
    for kind, (label, expected) in COHORT_INPUTS.items():
        path = root / label
        observed = ref(path)["sha256"] or "missing_file"
        check = gate(
            str(path),
            "sha256",
            expected,
            observed,
            "original_" + kind,
            "exp9110",
            ref(path)["sha256"],
        )
        checks.append(check)
        if check["passed"]:
            cohort[kind] = retain(path, raw, references)
    units: list[Json] = []
    if len(cohort) == 2:
        groups = cohort["roles"]["partition"]["roles"]["test"][:20]
        units = [
            dict(unit_id=r["unit_id"], qid=r["qid"], group=r["cluster"])
            for r in cohort["manifest"]["validated_rows"]
            if r["cluster"] in groups
        ]
        checks.append(
            gate(
                str(raw),
                "cohort_cardinality",
                [20, 60, 60],
                [len({r["qid"] for r in units}), len(units), len({r["unit_id"] for r in units})],
                "cohort",
                TASK,
            )
        )
    eligibility = dict(
        embedding=available[9112, "embeddings_ready_score"]
        and available[9117, "static_predictions_ready_score"],
        energy=available[9121, "energy_audit_ready_score"],
        online=available[9118, "online_predictions_ready_score"],
    )
    progress("before_cached_model_resolution_no_load")
    model = cached_current_model(gpu_index=0)
    progress("after_cached_model_resolution_no_load")
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(scope="pre-load inventory only; no service timing"),
        inference_substrate_class="blocked_no_run",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    work = dict(
        checks=checks,
        references=references,
        units=units,
        eligibility=eligibility,
        protocol=PROTOCOL,
        cached_model=model,
        current_work_receipt=receipt,
        duration_s=(end - start) / 1e9,
        historical_model_provenance=[
            dict(
                experiment_id=n,
                model_invocation_counts=v.get("model_invocation_counts", "missing_field"),
            )
            for n, v in values.items()
        ],
        raw=str(raw),
    )
    atomic_json(raw / "work.json", work)
    progress("after_preconditions")
    return dict(work, work_reference=ref(raw / "work.json"))


def build(work: Json, validation: Json, date: str) -> Json:
    """Publish missing evidence as a terminal block; never invent qualified service work."""
    if any(work["eligibility"].values()):
        raise ValueError("qualified_service_requires_measurement; inventory cannot price a service")
    for reference in work["references"]:
        if ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
            raise ValueError("retained_hash_drift")
    owned = validation["passed"] is True
    rows = [
        cost_row(
            u["unit_id"],
            9122,
            "complete_service",
            "unstarted",
            "science",
            arm=arm,
            group=u["group"],
            qid=u["qid"],
            original_unit_id=u["unit_id"],
            seed=SEED,
            family="LPHB",
            censor_reason="missing authenticated runtime/head service artifacts",
            exclusion_reason="external inputs unavailable; no load authorized",
        )
        for u in work["units"]
        for arm in ("embedding", "energy")
    ]
    checks = work["checks"]
    rows += [
        cost_row(
            "gate/" + str(index),
            9122,
            "evidence_check",
            "completed",
            "administrative",
            absolute_metric=int(check["passed"]),
            numerator=int(check["passed"]),
            denominator=1,
            metric="operand_check_passed",
            gate_operand=check,
            measurement_boundary="Boolean evidence observation; no service clock",
        )
        for index, check in enumerate(checks)
    ]
    failed = [r for r in checks if not r["passed"]]
    value: Json = dict(
        schema="carnot.v777_complete_cost.v1",
        experiment_id=9122,
        task_id=TASK,
        milestone="2026.11.777",
        run_date=date,
        honest_verdict="complete_blocked_" + failed[0]["check"]
        if owned
        else "complete_disqualified_owned_checks",
        verdict_class="blocked" if owned else "disqualified",
        status="blocked" if owned else "error",
        gate_check_summary=checks,
        rows=rows,
        protocol=work["protocol"],
        branch_eligibility=work["eligibility"],
        selected_units=work["units"],
        cost_inventory_ready_score=int(owned and not work.get("fixture_only", False)),
        fixture_only=work.get("fixture_only", False),
        embedding_service_ready_score=0,
        energy_service_ready_score=0,
        online_cost_ready_score=0,
        per_request_timing_rows=[],
        paired_service_cost_rows=[],
        complete_adaptation_rows=[],
        model_load_rows=[],
        failed_censored_work=[r for r in rows if r["status"] != "completed"],
        intended_count=len(rows),
        attempted_count=len(checks),
        completed_count=len(checks),
        failed_count=0,
        censored_count=0,
        unstarted_count=len(rows) - len(checks),
        excluded_count=0,
        effective_independent_groups=0,
        independent_count=0,
        sample_size_budget=dict(
            qids=20,
            answers=60,
            process_repetitions=3,
            selected_qids=len({u["qid"] for u in work["units"]}),
            selected_answers=len(work["units"]),
            scope="repetitions and siblings are not new independent units",
        ),
        scientific_request_counts=dict(
            intended=60 * 3 * 2,
            attempted=0,
            completed=0,
            failed=0,
            censored=0,
            unstarted=60 * 3 * 2,
        ),
        clock_boundary_map=[
            dict(
                operation=k,
                measured_seconds=None,
                nonoverlapping_fraction=None,
                status="unmeasured",
                reason="no qualified complete service clocks",
            )
            for k in (
                "cold_load",
                "raw_text_tokenization",
                "frozen_Qwen_forward",
                "transform_and_heads",
                "serialization",
                *PROTOCOL["adaptation"],
            )
        ],
        hardware_operation_rows=[
            dict(
                device=board,
                prerequisite=condition,
                measured_seconds=None,
                measured_fraction=None,
                physical_probe_performed=False,
            )
            for board, condition in CONDITIONS.items()
        ],
        hardware_scope=dict(
            CPU="counter, serialization, full CAL, storage and recovery",
            GPU="batched frozen features; requires GPU0 lease and real CUDA offload",
            FPGA="fixed-basis coefficient work is a future target only",
            KV260="historical quadratic only; k_max<=5",
            PolarFire="CPU-only",
            GateMate="unchanged physical/JTAG0xffffffff block",
            NPU="unqualified",
            TSU="unqualified; no supported sampling lowering",
        ),
        amdahl_ceiling_or_null=None,
        energy_joules_or_null=None,
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Administrative cost inventory only; complete services and adaptation unmeasured. No human benefit or ARC solve.",
        exposure_scope="LPHB exposed development; fixtures establish mechanics only. Exact-oracle successes would be circular_positive.",
        MODEL_SPECS=MODEL_SPECS,
        requested_inference_substrate_class="model_load_no_generation",
        inference_substrate_class="blocked_no_run",
        inference_substrate="aggregation_from_upstream_artifacts",
        current_llm_invocations=0,
        model_invoked=False,
        generated_tokens=0,
        model_invocation_counts=work["current_work_receipt"]["invocation_counts"],
        cached_model_resolution=work["cached_model"],
        execution_venue="host",
        historical_model_provenance=work["historical_model_provenance"],
        duration_s=work["duration_s"],
        random_seed=SEED,
        source_references=work["references"],
        current_work_receipts=[work["current_work_receipt"]],
        work_reference=work["work_reference"],
        code_config_hashes=work.get("sources", []),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_checks=owned,
            inventory_complete=owned,
            complete_service=False,
            embedding=False,
            energy=False,
            online=False,
        ),
        validation_scope_manifest=work.get("validation_scope_manifest", []),
        repository_health=validation.get("repository_health", {}),
        methodology_note="Authenticate exact current primary paths and original TEST snapshots before dependent work. Freeze the first20 original TEST groups/all60 siblings, three process repetitions, seed776108 and parity tolerances before outcomes. Raw-text embedding and teacher-forced energy protocols remain separate. Neither scientific benefit nor inventory elapsed time qualifies a service. All unavailable operands stay null; no model, board, download, alternative quantization or training work is attempted.",
    )
    value["field_principles"] = {
        k: "Retain exact operands and scope so missing evidence cannot become measured cost."
        for k in value
    }
    value["field_principles"].update(
        requested_inference_substrate_class="Intention is separate from actual executed operations.",
        inference_substrate_class="A pre-load refusal is blocked_no_run and does not owe a padded duration.",
        model_invocation_counts="Only owned attempted operations count; historical activity stays separate.",
        sample_size_budget="Sibling answers and repeated processes do not enlarge independent support.",
        cost_inventory_ready_score="Accounting readiness does not imply scientific service readiness.",
        embedding_service_ready_score="Hidden-state qualification is required independently of logits.",
        energy_service_ready_score="Energy remains an independent branch; no embedding or benefit gate.",
        online_cost_ready_score="Missing recovery or guard clocks cannot erase static costs.",
        amdahl_ceiling_or_null="Only measured disjoint full-service fractions justify ceilings.",
        energy_joules_or_null="Joules require sensor integrals; vendor TDP is not measured energy.",
        current_work_receipts="Monotonic inventory time is administrative and cannot price service work.",
        reproducibility_checksum="Bind all published fields, including failed operands.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute from retained rows and logs so rehashed summaries cannot invent work."""
    progress("before_cold_replay")
    try:
        value = json.loads(path.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            raise ValueError("checksum_drift")
        work = read_reference(value["work_reference"])
        work["work_reference"] = value["work_reference"]
        for reference in value["code_config_hashes"]:
            if ref(Path(reference["path"]))["sha256"] != reference["sha256"]:
                raise ValueError("code_hash_drift")
        for receipt in value["validation_receipts"]:
            log = ROOT / receipt["log_path"]
            if ref(log)["sha256"] != receipt["log_sha256"]:
                raise ValueError("validation_log_drift")
        validation = dict(
            passed=value["required_checks_passed"],
            receipts=value["validation_receipts"],
            coverage=value["owned_statement_coverage"],
            repository_health=value["repository_health"],
        )
        expected = build(work, validation, value["run_date"])
        progress("after_cold_replay")
        return dict(passed=expected == value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("cold_replay_refused")
        return dict(passed=False, reason=str(error))
