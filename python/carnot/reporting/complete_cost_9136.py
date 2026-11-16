"""REQ-REPORT-9136-V778: inventory complete costs without inventing service work.

Missing scientific inputs stop this invocation before a load. Original producer
pins and retained bytes make that refusal replayable after the checkout changes.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import sys
from types import ModuleType
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair
from carnot.reporting.complete_cost_9122 import (
    COHORT_INPUTS,
    PROTOCOL as PRIOR_PROTOCOL,
    checksum as checksum,
    read_reference,
    retain,
)
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt
from carnot.reporting.hardware_boundary_9079 import CONDITIONS
from carnot.reporting.service_cost_9093 import cost_row, gate, ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, SEED = "experiment_9136_v778_complete_cost", "exp9136-complete-cost", 776108
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/complete_cost_9136.py",
    "python/carnot/reporting/complete_cost_runner_9136.py",
    CLI,
]
TESTS = ["tests/python/test_complete_cost_9136.py"]
MODEL_SPECS = [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
INPUTS = {
    9124: (
        "results/experiment_9124_v778_contract.json",
        ["contract_ready_score", "source_ready_score"],
    ),
    9125: ("results/experiment_9125_v778_receipt_qualification.json", ["receipt_ready_score"]),
    9127: (
        "results/experiment_9127_v778_logit_runtime.json",
        ["logit_runtime_ready_score", "shard_feasible_score", "cost_feasible_score"],
    ),
    9130: ("results/experiment_9130_v778_logit_admission.json", ["admission_ready_score"]),
    9131: ("results/experiment_9131_v778_static_energy.json", ["static_predictions_ready_score"]),
    9133: ("results/experiment_9133_v778_delayed_streams.json", ["online_predictions_ready_score"]),
}
PINS = {
    9124: "sha256:a93fa7d34b6b6f151bcf5a7bc8c3fe9fcc1813395d68f9c825fba6b40ed0a4e2",
    9125: "sha256:77b9e1c0d194e269c5c3c6617a68f61e75624e6e0376ec2717a06291c857309d",
}
PROTOCOL = dict(
    PRIOR_PROTOCOL,
    process_arm_order=[["static", "online"] for _ in range(3)],
    static_arms=["prevalence", "probability_length", "raw_energy", "combined", "combined_donor"],
    online_arms=["frozen", "bias", "linear", "spline", "donor", "temperature"],
    model_admission=[
        "cached_sota_pair(gpu_indices=(0,0))",
        "GPU0 lease",
        "CUDA loader",
        "real offload",
        "GGUF embedded tokenizer",
        "model/build SHA256",
    ],
    cache_conditions=["cold_load", "warm_uncached", "cached_head_only"],
    exclusions=[],
    formulas="Exp9126 immutable full-vocabulary formulas; no gate or denominator changes",
)


def progress(phase: str) -> None:
    """Flush every boundary so a blocked input cannot look like a stalled model."""
    print(f"[exp9136] phase={phase}", flush=True)


def eligibility(available: dict[int, bool]) -> dict[str, bool]:
    """Allow static service independently of the separate online learning branch."""
    common = all(available.get(n, False) for n in (9124, 9125, 9127, 9130))
    return dict(
        static=common and available.get(9131, False), online=common and available.get(9133, False)
    )


def collect(root: Path, raw: Path, *, producer: ModuleType | None = None) -> Json:
    """Retain each observed operand once before deciding whether any load is eligible."""
    config = producer or sys.modules[__name__]
    config.progress("before_preconditions")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_json(raw / "protocol.json", config.PROTOCOL)
    checks: list[Json] = []
    references: list[Json] = []
    available = {}
    for identity, (label, fields) in config.INPUTS.items():
        path = root / label
        digest = ref(path)["sha256"]
        before = len(checks)
        try:
            value = retain(path, raw, references) if digest else {}
        except (ValueError, TypeError):
            value = {}
            checks.append(
                gate(
                    str(path), "JSON_object", True, "invalid_json", "input_format", identity, digest
                )
            )
        if digest:
            for field, expected in [("experiment_id", identity), ("required_checks_passed", True)]:
                checks.append(
                    gate(
                        str(path),
                        field,
                        expected,
                        value.get(field, "missing_field"),
                        "identity" if field == "experiment_id" else "upstream_validation",
                        identity,
                        digest,
                    )
                )
            if identity in config.PINS:
                checks.append(
                    gate(
                        str(path),
                        "sha256",
                        config.PINS[identity],
                        digest,
                        "immutable_primary",
                        identity,
                        digest,
                    )
                )
        for binding in value.get("code_config_hashes", []) + value.get("raw_shard_hashes", []):
            original = Path(binding["path"])
            original = original if original.is_absolute() else root / original
            observed = ref(original)["sha256"]
            checks.append(
                gate(
                    str(original),
                    "sha256",
                    binding["sha256"],
                    observed or "missing_file",
                    "producer_binding",
                    identity,
                    observed,
                )
            )
        for field in fields:
            checks.append(
                gate(
                    str(path),
                    field,
                    1,
                    value.get(field, "missing_field") if digest else "missing_file",
                    "upstream_readiness",
                    identity,
                    digest,
                )
            )
        available[identity] = all(c["passed"] for c in checks[before:])
        config.progress(f"after_source_{identity}")
    cohort: dict[str, Json] = {}
    for kind, (label, expected) in config.COHORT_INPUTS.items():
        path = root / label
        observed = ref(path)["sha256"]
        check = gate(
            str(path),
            "sha256",
            expected,
            observed or "missing_file",
            "original_" + kind,
            "exp9110",
            observed,
        )
        checks.append(check)
        if check["passed"]:
            cohort[kind] = retain(path, raw, references)
    units = []
    if len(cohort) == 2:
        groups = cohort["roles"]["partition"]["roles"]["test"][:20]
        units = [
            dict(
                unit_id=r["unit_id"],
                qid=r["qid"],
                group=r["cluster"],
                sibling=r.get("generator", r["unit_id"]),
            )
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
                config.TASK,
            )
        )
    disk = shutil.disk_usage(raw).free
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    for field, expected, observed in [
        ("free_disk_at_least_2GiB", True, disk >= 2 * 1024**3),
        (
            "address_space_at_least_256MiB",
            True,
            limit == resource.RLIM_INFINITY or limit >= 256 * 1024**2,
        ),
        ("llama_cpp_import_available", True, importlib.util.find_spec("llama_cpp") is not None),
    ]:
        checks.append(
            gate(str(raw), field, expected, observed, "resource_precondition", config.TASK)
        )
    config.progress("before_cached_model_resolution_no_load")
    pair = config.cached_sota_pair(gpu_indices=(0, 0)) or []
    model = next((s for s in pair if s["hf_id"] == config.MODEL_SPECS[0]["hf_id"]), None)
    config.progress("after_cached_model_resolution_no_load")
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(scope="pre-load inventory only; no service timings"),
        inference_substrate_class="blocked_no_run",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    work = dict(
        checks=checks,
        references=references,
        units=units,
        eligibility=config.eligibility(available),
        protocol=config.PROTOCOL,
        cached_model=model,
        current_work_receipt=receipt,
        duration_s=(end - start) / 1e9,
        raw=str(raw),
        resources=dict(free_disk_bytes=disk, address_space_limit=limit),
    )
    atomic_json(raw / "work.json", work)
    config.progress("after_preconditions")
    return dict(work, work_reference=ref(raw / "work.json"))


def build(work: Json, validation: Json, date: str) -> Json:
    """Reduce evidence rows; an administrative inventory cannot price eligible science."""
    if any(work["eligibility"].values()):
        raise ValueError("qualified_service_requires_measurement")
    for reference in work["references"]:
        read_reference(reference)
    owned = validation["passed"] is True
    rows = [
        cost_row(
            f"{u['unit_id']}/{arm}/{process}/{condition}",
            9136,
            "complete_service",
            "unstarted",
            "science",
            qid=u["qid"],
            group=u["group"],
            sibling=u["sibling"],
            original_unit_id=u["unit_id"],
            arm=arm,
            condition=condition,
            process=process,
            seed=SEED,
            censor_reason="external_inputs_unavailable",
            exclusion_reason="no authenticated complete scientific service",
        )
        for u in work["units"]
        for arm in ("static", "online")
        for process in range(3)
        for condition in ("warm_uncached", "cached_head_only")
    ]
    checks = work["checks"]
    rows += [
        cost_row(
            f"gate/{i}",
            9136,
            "evidence_check",
            "completed",
            "administrative",
            qid=None,
            sibling=None,
            seed=SEED,
            absolute_metric=int(c["passed"]),
            numerator=int(c["passed"]),
            denominator=1,
            metric="operand_check_passed",
            gate_operand=c,
            measurement_boundary="Boolean evidence observation; no service clock",
        )
        for i, c in enumerate(checks)
    ]
    value: Json = dict(
        schema="carnot.v778_complete_cost.v1",
        experiment_id=9136,
        task_id=TASK,
        milestone="2026.11.778",
        run_date=date,
        honest_verdict="complete_blocked_upstream_readiness"
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
        service_cost_ready_score=0,
        adaptation_cost_ready_score=0,
        fixture_only=work.get("fixture_only", False),
        paired_service_cost_rows=[],
        scientific_parity_rows=[],
        complete_adaptation_rows=[],
        cold_load_rows=[],
        per_request_timing_rows=[],
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
            scope="siblings and fresh processes do not increase independent support",
        ),
        scientific_request_counts=dict(
            intended=720, attempted=0, completed=0, failed=0, censored=0, unstarted=720
        ),
        clock_boundary_map=[
            dict(
                operation=k,
                measured_seconds=None,
                nonoverlapping_fraction=None,
                status="unmeasured",
                reason="no qualified complete service clock",
            )
            for k in (
                "cold_load",
                "raw_text_tokenization",
                "full_forward_reduction",
                "transform_and_all_heads",
                "typed_decisions_serialization",
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
            CPU="tokenization, transformations, heads, full CAL, serialization, fsync and recovery",
            GPU="frozen forward/reduction; GPU0 lease and actual CUDA offload required",
            FPGA="future fixed-basis coefficient work only",
            KV260="k_max<=5",
            PolarFire="CPU-only",
            GateMate="physical/JTAG0xffffffff block",
            NPU="unqualified",
            TSU="unqualified",
        ),
        amdahl_ceiling_or_null=None,
        energy_joules_or_null=None,
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Administrative availability inventory only; no measured service or adaptation cost, independent verifier benefit or ARC level solve. Does not close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208.",
        exposure_scope="Original LPHB exposed human development panel. Fixtures are mechanics only; exact-oracle gains would be circular_positive.",
        MODEL_SPECS=MODEL_SPECS,
        requested_inference_substrate_class="model_load_no_generation",
        inference_substrate_class="blocked_no_run",
        inference_substrate="aggregation_from_upstream_artifacts",
        execution_venue="host",
        current_llm_invocations=0,
        model_invoked=False,
        generated_tokens=0,
        forward_calls_attempted=0,
        forward_calls_completed=0,
        forward_input_tokens=0,
        model_invocation_counts=dict(
            work["current_work_receipt"]["invocation_counts"],
            forward_calls_attempted=0,
            forward_calls_completed=0,
            forward_calls_failed=0,
        ),
        cached_model_resolution=work["cached_model"],
        tokenizer_source="GGUF embedded tokenizer; not loaded",
        actual_offload=None,
        build_model_hashes=None,
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
            complete_adaptation=False,
        ),
        validation_scope_manifest=work.get("validation_scope_manifest", []),
        repository_health=validation.get("repository_health", {}),
        methodology_note="Freeze exact upstream fields and producer bytes before model work. Retain first20 original TEST groups/all60 siblings, three fresh processes, seed776108 and parity tolerances. Static and online gates are independent. Missing scientific inputs terminate before load; inventory time is not latency. No duration padding, fallback, downloads, generation, training, GPU1, board contact, Rust speedup or assumed learning gain.",
    )
    value["field_principles"] = dict(
        gate_check_summary="Exact operands distinguish missing files, missing fields and observed zero.",
        rows="Unstarted scientific work remains visible beside completed administrative checks.",
        branch_eligibility="Missing learning trajectories cannot independently block qualified static service.",
        cost_inventory_ready_score="Complete accounting does not establish service performance.",
        service_cost_ready_score="Readiness requires complete matched outer clocks and scientific parity.",
        adaptation_cost_ready_score="Sparse updates cannot replace full guards, durable commits and actual recovery.",
        paired_service_cost_rows="Outer clocks must include tokenization, forward, all eligible heads and serialization.",
        scientific_parity_rows="A faster path must preserve probabilities within frozen tolerance and identical decisions.",
        complete_adaptation_rows="Rejected proposals, failed work, final drain and recovery all have a cost.",
        cold_load_rows="Fresh process load time stays separate from warm request latency.",
        failed_censored_work="Unavailable or failed work cannot vanish from the denominator.",
        sample_size_budget="Siblings and repeated processes do not create independent questions.",
        clock_boundary_map="Only matching nonoverlapping measured clocks support fractions.",
        hardware_operation_rows="Future operation mapping is separate from measured board performance.",
        amdahl_ceiling_or_null="Amdahl ceilings require measured component fractions.",
        energy_joules_or_null="Joules require power integrals, never vendor TDP.",
        MODEL_SPECS="The mandated model cannot be silently replaced.",
        requested_inference_substrate_class="Intended work stays distinct from executed work.",
        inference_substrate_class="A pre-load refusal is blocked_no_run and owes no padded duration.",
        current_work_receipts="Only owned current events count as executed model work.",
        model_invocation_counts="Attempted forward work is not generated tokens or historical inference.",
        source_references="Retained immutable bytes preserve historical producer authority.",
        validation_receipts="Readiness requires real child exits and byte-bound checker logs.",
        independent_generalization_score="Exposed development cannot establish independent human generalization.",
        reproducibility_checksum="All published observations, including failed gates, are bound together.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path, *, producer: ModuleType | None = None) -> Json:
    """Rebuild the terminal artifact from retained work so rehashed invention fails."""
    config = producer or sys.modules[__name__]
    config.progress("before_cold_replay")
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
            if ref(config.ROOT / receipt["log_path"])["sha256"] != receipt["log_sha256"]:
                raise ValueError("validation_log_drift")
        validation = dict(
            passed=value["required_checks_passed"],
            receipts=value["validation_receipts"],
            coverage=value["owned_statement_coverage"],
            repository_health=value["repository_health"],
        )
        expected = config.build(work, validation, value["run_date"])
        config.progress("after_cold_replay")
        return dict(passed=expected == value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        config.progress("cold_replay_refused")
        return dict(passed=False, reason=str(error))
