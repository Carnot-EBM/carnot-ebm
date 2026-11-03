"""Freeze comparable verifier inputs so later studies cannot tune them after scoring."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.reporting import verifier_completeness_8715 as old
from carnot.reporting import v748_collection_contract as binding
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8726_v748_verification_methods", "exp8726-verification-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/verification_methods{s}_8726.py" for s in ("", "_runner")] + [
    CLI
]
TESTS = ["tests/python/test_verification_methods_8726.py"]
MODEL_SPECS: list[Json] = []
BASELINE = old.BASELINE
TARGETED = ("truth", "invalid", "entity_bijection", "required_premise", "required_role")
SCHEDULES = dict(baseline=BASELINE, targeted=TARGETED)
authority, upstream = binding.m, old.upstream
reference, load, checksum = old.reference, old.load, old.checksum
LITERATURE = f"results/raw/{NAME}/literature/retrievals.json"
LITERATURE_PIN = "sha256:3c1fc96eee7caca11fafb6770113954ed4bd6c1161024d97f2848a03d6b54f64"
PRODUCER = "results/experiment_8715_v747_verifier_completeness.json"
PIN = "sha256:1e205986c906fecbb1578c0df30ea828a31d7c8b42c8cfd736868f98bd7965b2"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries because an unresponsive child must remain distinguishable from work."""
    print(f"[exp8726] phase={phase} completed={completed} pending={pending}", flush=True)


def inputs(root: Path) -> Json:
    """Require original producer acceptance independently of other branch readiness."""
    result = old.inputs(root)
    result["references"].append(dict(binding.PRESERVED))
    receipt = upstream.custody(
        dict(path=str(root / PRODUCER), sha256=PIN), "verifier_audit_ready_score", 8715
    )
    result["references"] += receipt["references"]
    result["failures"] += receipt["gate_check_summary"]
    try:
        original = load(dict(path=str(root / PRODUCER), sha256=PIN))
        primitive = load(original["primitive_reference"])
        if primitive["frozen"] != old.freeze(result["worlds"]):
            raise ValueError("original_eleven_transforms_or_split_drift")
        result["references"].append(original["primitive_reference"])
        path = root / LITERATURE
        if reference(path)["sha256"] != LITERATURE_PIN:
            raise ValueError("literature_retrieval_identity")
        retrievals = json.loads(path.read_bytes())
        result["references"].append(reference(path))
        result["references"] += [
            dict(path=r["path"], sha256=r["sha256"])
            for r in retrievals
            if r["status"] == "retrieved"
        ]
        result["retrievals"] = retrievals
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["failures"].append(
            dict(
                authority.gate(
                    root / PRODUCER, "original_protocol_and_literature", True, str(error), TASK
                ),
                check="original_protocol_and_literature",
            )
        )
        result["retrievals"] = []
    result["failures"] += [
        g for ref in result["references"] if not (g := upstream.authenticate(ref))["passed"]
    ]
    result["passed"] = not result["failures"]
    return result


def freeze(worlds: list[Json]) -> Json:
    """Bind prompts to explicit premises; removing evidence must not turn unknown into false."""
    frozen = old.freeze(worlds)
    for world in frozen["worlds"]:
        world["original_partition"] = world.pop("partition")
        world["exposure_scope"] = "exposed-development"
    pilot = []
    for family in upstream.FAMILIES:
        for world in sorted(
            (w for w in worlds if w["family"] == family), key=lambda w: w["world_id"]
        )[:4]:
            views = {v["view"]: v for v in world["views"]}
            for name, original in [
                ("supported", "supported"),
                ("contradicted", "contradicted"),
                ("equivalent", "equivalent_wording"),
                ("missing_premise", "supported"),
            ]:
                view = views[original]
                query = view["query"] if isinstance(view["query"][0], list) else [view["query"]]
                facts = deepcopy(world["facts"])
                if name == "missing_premise":
                    facts = [f for f in facts if f[:2] != query[-1][:2]]
                evidence = dict(entities=world["entities"], family=family, facts=facts)
                prompt = json.dumps(
                    dict(
                        evidence=evidence,
                        proposed_answer=dict(text=view["claim"], required_predicates=query),
                    ),
                    sort_keys=True,
                )
                prompt += "\nDoes the evidence SUFFICIENTLY SUPPORT the proposed answer? Use only explicit premises. Answer only Yes or No.\nAnswer:"
                truth = old.oracle(facts, query)
                pilot.append(
                    dict(
                        prompt_id=world["world_id"] + "/" + name,
                        world_id=world["world_id"],
                        family=family,
                        view=name,
                        evidence=evidence,
                        query=query,
                        proposed_answer=view["claim"],
                        prompt=prompt,
                        prompt_sha256=canonical_hash(prompt),
                        truth=truth,
                        support_label=int(truth == 1),
                        source_world_sha256=canonical_hash(world),
                        status="frozen_unscored",
                        p_yes=None,
                    )
                )
    return dict(frozen, schedules={k: list(v) for k, v in SCHEDULES.items()}, pilot=pilot)


def p_yes(yes: float | None, no: float | None) -> float:
    """Normalize observed first-token log probabilities without inventing confidence."""
    if yes is None or no is None or not math.isfinite(yes) or not math.isfinite(no):
        raise ValueError("first_token_log_probabilities_required")
    maximum = max(yes, no)
    return math.exp(yes - maximum) / (math.exp(yes - maximum) + math.exp(no - maximum))


def measure(frozen: Json, raw: Path) -> Json:
    """Execute every scheduled call; equal invocation counts do not imply equal runtime."""
    rows = []
    for index, world in enumerate(frozen["worlds"]):
        probes = {p["transform"]: p for p in world["probes"]}
        for variant in frozen["variants"]:
            for arm, schedule in frozen["schedules"].items():
                local = []
                start = time.monotonic_ns()
                for transform in schedule:
                    probe = probes[transform]
                    actual = old.verifier(variant, probe["facts"], probe["query"])
                    local.append(
                        dict(
                            transform=transform,
                            actual=actual,
                            expected=probe["expected"],
                            query_predicates_supplied=len(probe["query"] or []),
                        )
                    )
                elapsed = (time.monotonic_ns() - start) / 1e9
                passed = all(p["actual"] == p["expected"] for p in local)
                rows.append(
                    dict(
                        world_id=world["world_id"],
                        original_partition=world["original_partition"],
                        exposure_scope="exposed-development",
                        variant=variant,
                        arm=arm,
                        probes=local,
                        variant_invocations=len(local),
                        predicate_work=sum(p["query_predicates_supplied"] for p in local),
                        latency_s=elapsed,
                        passed=passed,
                        raw_numerator=sum(p["actual"] == p["expected"] for p in local),
                        raw_denominator=5,
                        status="completed",
                        absolute_metric=int(passed),
                        censored=False,
                        excluded=False,
                    )
                )
        progress("world_controls", index + 1, len(frozen["worlds"]) - index - 1)
    atomic_json(raw / "probes.json", dict(rows=rows))
    return dict(rows=rows)


def build(work: Json, output: Path) -> Json:
    """Keep CPU readiness separate from missing CUDA and from constructed scientific gains."""
    validation = work["validation"]
    checked = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["coverage"]["complete"]
        and validation["custody"]["passed"]
    )
    external = work["inputs"]["passed"] and work["contract"]["ready"] and work["resources"]["ready"]
    rows = deepcopy(work["measurement"]["rows"])
    complete = (
        len(rows) == 1440
        and len(work["frozen"]["pilot"]) == 48
        and all(r["variant_invocations"] == 5 for r in rows)
    )
    ready = int(
        checked
        and external
        and complete
        and not work["private_control"]
        and not work["adversarial_findings"]
    )
    verdict = (
        "blocked"
        if not external
        else "disqualified"
        if not work["private_control"] and (not checked or work["adversarial_findings"])
        else "circular_positive"
        if ready
        else "null"
    )
    gates = deepcopy(work["inputs"]["failures"] + work["contract"]["gate_check_summary"])
    for field, passed in [
        ("disk_memory_scratch", work["resources"]["ready"]),
        ("complete_protocol", complete),
    ]:
        if not passed:
            gates.append(authority.gate(Path(work["raw"]), field, True, passed, TASK))
    for row in gates:
        row.setdefault("check", row.get("artifact_field", "operand"))
    completed = len(rows)
    if not rows:
        rows = [
            dict(
                world_id=f"{family}:{index:02d}",
                variant=variant,
                arm=arm,
                status="blocked_external_operand",
                variant_invocations=0,
                latency_s=None,
                predicate_work=None,
                absolute_metric=0,
                censored=True,
                excluded=False,
            )
            for family in upstream.FAMILIES
            for index in range(40)
            for variant in old.VARIANTS
            for arm in SCHEDULES
        ]
    methods = [
        dict(
            arxiv="2610.25474",
            method="Normalize observed first-token Yes/No log probabilities; fixed candidate support, no search",
            test="test_protocol",
            code_status="author verifier.py retrieved; HTML endpoint failed, author full methods retrieved",
        ),
        dict(
            arxiv="2610.28679",
            method="Valid alternatives and invalid cases; equal five-call budgets",
            test="test_protocol",
            code_status="author project says Code soon; no code release link available",
        ),
        dict(
            arxiv="2608.00585",
            method="Require every premise; preserve unknown separately from false",
            test="test_protocol",
            code_status="linked author repository rechecked",
        ),
        dict(
            arxiv="2602.02056v4",
            method="Report local predicate inputs and time separately; no transfer of FPGA speed",
            test="test_protocol",
            code_status="no author code link identified in primary text",
        ),
    ]
    value = dict(
        experiment_id=8726,
        task_id=TASK,
        milestone="2026.11.748",
        run_date="20261102",
        invocation_id=work["invocation_id"],
        honest_verdict="complete_" + verdict + "_v748_verification_methods",
        verdict_class=verdict,
        gate_check_summary=gates,
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        rows=rows,
        intended_count=1440,
        completed_count=completed,
        failed_count=0,
        censored_count=1440 - completed,
        excluded_count=0,
        independent_count=120 if completed else 0,
        verifier_is_oracle=True,
        exposure_scope="All original120 worlds are exposed-development; retain original90 development/30 evaluation IDs; no independent generalization",
        independent_generalization_score=0,
        acceptance_gates=dict(
            original_operands=work["inputs"]["passed"],
            v748_task_binding=work["contract"]["ready"],
            resources=work["resources"]["ready"],
            complete_protocol=complete,
            owned_validation=checked,
        ),
        required_checks_passed=checked,
        flagged_adversarial=bool(
            work["adversarial_findings"] or (not work["private_control"] and not checked)
        ),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        retained_child_custody=validation["custody"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work["adversarial_findings"],
        preconditions_checked=work["resources"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7488726,
        source_artifact_hashes=work["inputs"]["references"] + work["contract"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=validation["raw_shard_hashes"] + work["raw_references"],
        cited_upstream_artifacts=work["inputs"]["references"],
        verification_methods_ready_score=ready,
        runtime_repair_ready_score=work["runtime"]["ready"],
        runtime_gate_check_summary=work["runtime"]["gate_check_summary"],
        runtime_evidence=work["runtime"],
        probe_schedule_rows=[
            dict(
                arm=k,
                ordered_transforms=list(v),
                budget=5,
                short_circuit=False,
                intended_variant_world_units=720,
                intended_variant_invocations=3600,
            )
            for k, v in SCHEDULES.items()
        ],
        qwen_pilot_roster=work["frozen"]["pilot"],
        literature_method_rows=methods,
        literature_retrievals=work["inputs"]["retrievals"],
        execution_authority=work["contract"],
        canonical_tasks_sha256=authority.PIN,
        qwen_pilot_configuration=dict(
            execution_owner="exp8730-qwen-self-score-canary",
            model="unsloth/Qwen3.8-27B-GGUF",
            quantization="Q4_K_M",
            temperature=0,
            max_output_tokens=8,
            prompts=48,
            total_token_cap=384,
            scoring="exp(logp_Yes-logsumexp(logp_Yes,logp_No))",
            missing_logits="blocked",
            metrics=["support_brier", "false_support", "equivalence", "high_confidence_errors"],
            decision_threshold=0.5,
            high_confidence_threshold=0.9,
            calls_this_invocation=0,
            runtime_prerequisite="causal qualified repair, context/copy and offload receipts matching runtime",
        ),
        predicate_work_definition="Count query predicates actually supplied to each variant; constant-return variants still receive these inputs. This is input volume, not internal comparisons or instruction counts.",
        probe_cost_comparison="Exactly five invocations per variant and arm; measured elapsed time remains separate. No equal-wall-time or superiority claim.",
        publication_output=str(output.absolute()),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        methodology_note="Freeze independently authenticated V748 schedules and48 source-bound support prompts. Execute constructed oracle controls with zero model calls. Preserve V747120/130 unequal-budget result; no natural extraction, causal CUDA repair, ARC outcome, learning benefit, independent generalization or physical acceleration follows. Exp8729 owns equal-budget utility; Exp8730 alone owns Qwen scores.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind exact evidence and actual invocation; protocol eligibility is separate from scientific benefit and optional CUDA readiness.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value
