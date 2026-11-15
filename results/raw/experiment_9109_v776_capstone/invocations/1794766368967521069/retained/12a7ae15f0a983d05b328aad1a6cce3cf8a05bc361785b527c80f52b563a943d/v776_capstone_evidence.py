"""REQ-REPORT-9109-V776: retained rows support accounting, never independent truth."""

from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import v775_capstone_evidence as history
from carnot.reporting import qwen_judge_9107 as judge
from carnot.reporting.current_work_receipt import canonical_hash as canonical_hash
from carnot.reporting.encoder_head_diagnosis_8972 import reference
from carnot.reporting.service_cost_9093 import gate
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v768_capstone import edge as edge

Json = dict[str, Any]
ROOT = history.ROOT
TASK_PIN = "sha256:22a0b8c4aea63e55dc48e0cf4ecba693f533504028d9578daefe8c20014350d1"
SUFFIXES = (
    "contract",
    "human-authority",
    "embedding-runtime",
    "online-methods",
    "embeddings-a",
    "embeddings-b",
    "static-heads",
    "online-streams",
    "static-audit",
    "learning-audit",
    "signal-controls",
    "qwen-judge",
    "complete-cost",
    "capstone",
)
TASKS = [f"exp{9096 + i}-{s}" for i, s in enumerate(SUFFIXES)]
PINS = {
    "results/experiment_9096_v776_contract.json": "sha256:71021947c3a5165e2575ae4611fc5d2b9108c5544c289ab9a84c8de37f0c1076",
    "results/experiment_9097_v776_human_authority.json": "sha256:24c9e0983d9a4eb0aa5a488b5467386e57d6f13b79447f0f3a2f3d1c68e25532",
    "results/experiment_9098_v776_embedding_runtime.json": "sha256:3eb28ae1a30060138bc99a7b46f85f5ac5a94a840a2c51996ac1c4a0e3f92dff",
    "results/experiment_9099_v776_online_methods.json": "sha256:ca82730179091d4a80d9a4c1384f04d0e38e24937c22b4277932b0dce7af34c7",
    "results/experiment_9100_embeddings_a.json": "sha256:a962ba36757869278a8f4a79b745a779990cb93fdc71eeee57dc889925880543",
    "results/experiment_9101_embeddings_b.json": "sha256:959e4bff68927057eafb27f4672772d9fbffd0b5e6a2c2929109bc79c7f0a57c",
    "results/experiment_9107_v776_qwen_judge.json": "sha256:1a8053cdf2fb2a5f111e97bfe888cdf4200d638131903d97e210dbdf993ac617",
    "results/experiment_9095_v775_capstone.json": "sha256:e3b8e64fb3aad034ab99a10a0985b07b55522dcc1606ec3b4b80bc73e950f100",
    "results/experiment_8435_v726_capacity_benefit_audit.json": "sha256:e2e58cbf680becad2f1572629337a58d747911492c24fa43697a8d867f4b0a51",
    "results/experiment_8972_v767_encoder_head_diagnosis.json": "sha256:aca5e51be78c64c911d49e6ce3cf3e036393ff59f1a300af29486f91733c56f4",
    **{
        p: history.PINS[p]
        for p in [
            "results/experiment_9084_v775_human_methods.json",
            "results/experiment_9088_human_embeddings.json",
            "results/experiment_9093_v775_service_cost.json",
        ]
    },
    "research-roadmap.yaml": "sha256:05c2129cbd2cba5a828860c221e1e459d337579167d97241d9626cb5c3fa0517",
    "ops/conductor-log.md": "sha256:16fc5dc55e482072b531301a932efa6036b7f9618cae4d34c4c99075611ac268",
    "research-complete.yaml": "sha256:934897c3f6dd5293ccf092b2880aea830d16ea8edbcee0eff9c3949cff554828",
    "ops/arc_solve_registry.yaml": "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
    "ops/exclusion_manifest.yaml": "sha256:081d0cb3f30a97c2afae61715d9c84de5254acae75eef050e31a3a221bdfc8a0",
}


def project(value: Json) -> Json:
    """Retain complete source bytes while keeping unused old arrays out of current work."""
    if value.get("experiment_id", value.get("experiment")) in {8435, 8972, 9084, 9088, 9093}:
        return {
            k: v
            for k, v in value.items()
            if k in {"experiment_id", "experiment", "task_id", "honest_verdict", "verdict_class"}
        }
    return history.project(value)


def check(path: Path, field: str, expected: Any, observed: Any, name: str) -> Json:
    """An absent operand must not become a measured zero or a passing null."""
    return gate(str(path), field, expected, observed, name, TASKS[-1], reference(path)["sha256"])


def authenticate(root: Path, raw: Path) -> Json:
    """Copy exact producer bindings before any reduction; never repair old sources."""
    checks, snapshots, values, operands = [], {}, {}, {}
    for name, pin in PINS.items():
        path = root / name
        checks.append(
            check(path, "sha256", pin, reference(path)["sha256"] or "missing_file", "source_bytes")
        )
        if checks[-1]["passed"]:
            snapshots[name] = retain(path, raw, pin)
    tasks = []
    for name, ref in snapshots.items():
        if not name.startswith("results/"):
            continue
        value = project(json.loads(Path(ref["path"]).read_bytes()))
        n = int(Path(name).name.split("_")[1])
        if n == 9096:
            tasks = value["full_task_objects"]
        task = TASKS[n - 9096] if 9096 <= n <= 9108 else f"prior:{n}"
        entry: Json = dict(value=value, reference=ref, selected_name=name, current_code_drift=[])
        values[task] = entry
        if not 9095 <= n <= 9108:
            continue
        scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
        valid = (
            value.get("experiment") == n
            if scheduler
            else (value.get("experiment_id") == n and value.get("task_id") == task)
        )
        checks.append(
            check(
                root / name,
                "producer_identity",
                True,
                valid if n != 9095 else value.get("task_id") == "exp9095-capstone",
                "producer_identity",
            )
        )
        bindings = [
            *value.get("raw_shard_hashes", []),
            *value.get("source_references", []),
            *value.get("current_work_receipts", []),
            *value.get("code_config_hashes", []),
        ]
        bindings += [
            r for k, r in value.items() if k.endswith("_reference") and isinstance(r, dict)
        ]
        for bound in bindings:
            if not bound.get("path") or not bound.get("sha256"):
                continue
            path = Path(bound["path"])
            # Model binaries are opaque historical provenance; no current inference consumes them.
            if path.suffix in {".gguf", ".so"} or str(path) in operands:
                continue
            row = check(
                path,
                "sha256",
                bound["sha256"],
                reference(path)["sha256"] or "missing_file",
                "producer_binding",
            )
            if row["passed"]:
                operands[str(path)] = retain(path, raw, bound["sha256"])
            elif bound in value.get("code_config_hashes", []):
                entry["current_code_drift"].append(row)
            else:
                checks.append(row)
        print(
            f"[exp9109] phase=producer_bindings_authenticated task={task} operands={len(operands)}",
            flush=True,
        )
    if tasks:
        checks.append(
            check(root, "canonical_task_hash", TASK_PIN, canonical_hash(tasks), "task_authority")
        )
        active = (
            yaml.safe_load(Path(snapshots["research-roadmap.yaml"]["path"]).read_bytes())
            if "research-roadmap.yaml" in snapshots
            else {}
        )
        checks.append(
            check(
                root / "research-roadmap.yaml",
                "full_task_objects",
                tasks,
                active.get("tasks", "missing_field"),
                "active_authority",
            )
        )
    return dict(checks=checks, snapshots=snapshots, values=values, operands=operands, tasks=tasks)


def primitive(work: Json, ref: Json) -> Json:
    """Only retained, byte-bound rows may feed a new aggregate."""
    saved = work["operands"][ref["path"]]
    require(saved["sha256"] == ref["sha256"], "primitive_identity")
    return json.loads(Path(saved["path"]).read_bytes())  # type: ignore[no-any-return]


def outcomes(work: Json) -> list[Json]:
    """Keep missing declared artifacts distinct from authentic scheduler alternatives."""
    logref = work["snapshots"].get("ops/conductor-log.md")
    lines = Path(logref["path"]).read_text().splitlines() if logref else []
    rows = []
    for i, identity in enumerate(TASKS[:-1]):
        task = (
            work["tasks"][i]
            if work["tasks"]
            else dict(
                id=identity,
                deliverable=f"results/experiment_{9096 + i}_v776_{SUFFIXES[i].replace('-', '_')}.json",
            )
        )
        entry = work["values"].get(identity, {})
        value = entry.get("value", {})
        scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
        primary = bool(entry and not scheduler)
        gate_rows = [edge(g, work["values"]) for g in task.get("gated_on", [])]
        gate_rows += [r for r in value.get("gate_check_summary", []) if isinstance(r, dict)]
        failed = [r for r in gate_rows if not r["passed"]]
        rows.append(
            dict(
                task_id=identity,
                experiment_id=9096 + i,
                task=task,
                primary_path=task["deliverable"],
                primary_status="present" if primary else "missing_file",
                reference=entry.get("reference"),
                selected_path=entry.get("selected_name"),
                scheduler_alternative=entry.get("reference") if scheduler else None,
                disposition="authenticated_scheduler_skip"
                if scheduler
                else "producer_primary"
                if primary
                else "unstarted_upstream_retired",
                scheduler_events=[
                    line for line in lines if task.get("title", "ABSENT")[:50].strip() in line
                ],
                honest_verdict=value.get("honest_verdict") if primary else None,
                accounting_honest_verdict="complete_blocked_declared_gate"
                if not primary
                else value.get("honest_verdict"),
                scheduler_honest_verdict=value.get("honest_verdict") if scheduler else None,
                verdict_class=value.get("verdict_class", "blocked"),
                readiness={k: v for k, v in value.items() if k.endswith("ready_score")},
                failed_gate_value=value.get(
                    "failed_observed",
                    failed[0]["observed"] if failed else "missing_file" if not primary else None,
                ),
                gate_check_summary=gate_rows,
                qualified=bool(
                    primary
                    and value.get("required_checks_passed")
                    and value.get("verdict_class") != "disqualified"
                ),
                current_code_drift=entry.get("current_code_drift", []),
            )
        )
    return rows


def retirement(work: Json) -> list[Json]:
    """A changed mechanism does not erase a prior null or expand its original scope."""
    result = []
    by_number = {
        int(t.split(":")[-1]): v for t, v in work["values"].items() if t.startswith("prior:")
    }
    for task in work["tasks"]:
        for prior in task["prior_failures"]:
            old = by_number.get(int(prior["experiment_id"][3:7]), {})
            current = work["values"].get(task["id"], {})
            actual = old.get("value", {}).get("honest_verdict")
            verdict = current.get("value", {}).get("honest_verdict")
            if task["id"] == TASKS[-1]:
                verdict = "complete_blocked_declared_gate"
            same = bool(prior["retire_if_same_verdict"] and actual == prior["verdict"] == verdict)
            result.append(
                dict(
                    task_id=task["id"],
                    scope=task["title"],
                    prior_task_id=prior["experiment_id"],
                    declared_prior_verdict=prior["verdict"],
                    actual_prior_verdict=actual,
                    prior_verdict_authenticated=actual == prior["verdict"],
                    prior_reference=old.get("reference"),
                    current_verdict=verdict,
                    current_reference=current.get("reference"),
                    retire_if_same_verdict=prior["retire_if_same_verdict"],
                    condition_met=same,
                    changed_mechanism_context=prior["addressed_by"],
                    recommendation="retire_unchanged_scientific_prerequisite"
                    if same
                    else "preserve_prior_scope_require_new_qualified_evidence",
                    accounting_continues_unconditionally=True,
                    manifest_mutated=False,
                )
            )
    return result


def reduce(work: Json) -> Json:
    """Recompute descriptive evidence while leaving unavailable science null."""
    require(canonical_hash(work["tasks"]) == TASK_PIN, "task_authority")
    values = {t: v["value"] for t, v in work["values"].items()}
    source, runtime, methods, fixed = [values[TASKS[i]] for i in [1, 2, 3, 11]]
    human_rows = primitive(work, source["primitive_reference"])["reduction"]["rows"]
    require(human_rows == source["rows"], "human_primitive_rows")
    require(len({r["unit_id"] for r in human_rows}) == len(human_rows), "human_unit_identity")
    human = dict(
        question_count=len({r["qid"] for r in human_rows}),
        annotator1_observed_count=sum(r["annotator1"] is not None for r in human_rows),
        annotator2_observed_count=sum(r["annotator2"] is not None for r in human_rows),
        annotator2_missing_count=sum(r["annotator2"] is None for r in human_rows),
        disagreement_count=sum(r["disagreement"] for r in human_rows),
        role_counts=dict(Counter(r["role"] for r in human_rows)),
        rows=human_rows,
        independent_eval_ready_score=0,
        claim_scope="Authenticated exposed LPHB inventory only.",
    )
    require(
        human["role_counts"] == dict(TRAIN=450, CAL=150, TEST=300)
        and human["question_count"] == 300,
        "original_partition",
    )
    rwork = primitive(work, runtime["primitive_reference"])
    runtime_rows = rwork["warm"]["rows"]
    require(runtime_rows == runtime["rows"], "runtime_primitive_rows")
    runtime_summary = dict(
        intended_count=12,
        completed_count=sum(r["status"] == "completed" for r in runtime_rows),
        embeddings_ready_score=runtime["embeddings_ready_score"],
        logits_ready_score=runtime["logits_ready_score"],
        embedding_pass_count=sum(r["embedding_passed"] for r in runtime_rows),
        logits_pass_count=sum(r["logits_passed"] for r in runtime_rows),
        scope="Twelve capability fixtures, no natural factuality result.",
    )
    mwork = primitive(work, methods["primitive_reference"])
    require(mwork["result"]["rows"] == methods["rows"], "methods_primitive_rows")
    jwork = primitive(work, fixed["primitive_reference"])
    jrows = primitive(work, fixed["judge_prediction_reference"])["rows"]
    require(jrows == fixed["rows"] == jwork["runtime"]["rows"], "judge_primitive_rows")
    scores = judge.reduce_scores(
        jrows, {r["unit_id"]: r["annotator1"] for r in human_rows}, draws=0
    )
    require(scores["metrics"] == fixed["metrics"], "judge_metrics")
    require(scores["test_predictions"] == fixed["test_predictions"], "judge_predictions")
    table = []
    for branch, index, prerequisite in [
        (
            "human_authority",
            1,
            "Genuinely new qualified source for independent human confirmation.",
        ),
        (
            "runtime",
            2,
            "A frozen Qwen runtime exposing valid hidden states for every nonempty fixture, serial/batch parity and real offload.",
        ),
        (
            "feature_shard_a",
            4,
            "embeddings_ready_score=1, then the original first150 qids and450 sibling responses.",
        ),
        (
            "feature_shard_b",
            5,
            "embeddings_ready_score=1, then the original remaining150 qids and450 sibling responses.",
        ),
        (
            "static_six_arm",
            6,
            "Both complete authenticated feature shards and the unchanged six-arm protocol.",
        ),
        (
            "causal_future_benefit",
            8,
            "Qualified representations, sealed delayed streams and original grouped future metrics.",
        ),
        (
            "retention",
            9,
            "Paired causal stream predictions, exact recovery and untouched original retention rows.",
        ),
        (
            "semantic_controls",
            10,
            "Qualified static heads and fixed question-only, answer-only, length and generator controls.",
        ),
        (
            "fixed_qwen_judge",
            11,
            "Correct the flagged producer provenance and pass unchanged artifact checks in a new result; preserve this disqualification.",
        ),
        (
            "complete_cost",
            12,
            "Matched qualified static/online/judge request clocks, durability, recovery and measured power integrals.",
        ),
    ]:
        entry = work["values"].get(TASKS[index], {})
        v = entry.get("value", {})
        table.append(
            dict(
                branch=branch,
                task_id=TASKS[index],
                source_reference=entry.get("reference"),
                verdict_class=v.get("verdict_class", "blocked"),
                metric=None,
                numerator=None,
                denominator=0,
                status="diagnostic_disqualified"
                if index == 11
                else "administrative_only"
                if index in [1, 2]
                else "unmeasured",
                next_prerequisite=prerequisite,
                scientific_readiness=0,
                independent_generalization_score=0,
            )
        )
    old = values["prior:9095"]
    historical_work = primitive(work, old["work_reference"])
    inherited = history.reduce(historical_work)
    bounds = {k: inherited[k] for k in ["seven_bound_rows", "ten_bound_rows"]}
    require(
        all(inherited[k] == old[k] for k in [*bounds, "cost_summary", "susceptibility_summary"]),
        "historical_aggregates",
    )
    require([len(bounds[k]) for k in bounds] == [7, 10], "original_bound_count")
    prior_null = values["prior:8435"]
    require(
        prior_null["honest_verdict"] == "complete_null_v726_capacity_benefit_audit",
        "prior_null_scope",
    )
    return dict(
        **bounds,
        human_summary=human,
        runtime_summary=runtime_summary,
        online_methods_summary=dict(
            row_count=len(methods["rows"]),
            scope="CPU method fixtures only",
            human_future_benefit=None,
            human_retention=None,
        ),
        judge_summary=dict(
            metrics=scores["metrics"],
            qid_rows=scores["qid_rows"],
            temperature=scores["temperature"],
            completed_count=len(jrows),
            effective_independent_groups=len(scores["qid_rows"]),
            qualified=False,
            verdict_class=fixed["verdict_class"],
            source_reference=work["values"][TASKS[11]]["reference"],
            flags=fixed.get("flagged_adversarial_provenance"),
            claim_scope="Disqualified exposed-development diagnostic; no accepted judge comparison.",
        ),
        evidence_table=table,
        static_six_arm_outcomes=[
            dict(
                arm=arm,
                condition="unmeasured",
                original_qid=None,
                seed=None,
                absolute_metric=None,
                numerator=None,
                denominator=0,
                status="censored",
                censor_reason="missing_complete_feature_shards",
            )
            for arm in primitive(work, source["static_protocol_reference"])["arms"]
        ],
        causal_future_benefit=None,
        calibration_retention=None,
        semantic_control_outcomes=None,
        full_service_cost=None,
        cost_summary=inherited["cost_summary"],
        susceptibility_summary=inherited["susceptibility_summary"],
        historical_reduction_reference=work["operands"][old["work_reference"]["path"]],
        preserved_cpu_reference=work["values"]["prior:9095"]["reference"],
        original_exp8435_null=dict(
            reference=work["values"]["prior:8435"]["reference"],
            honest_verdict=prior_null["honest_verdict"],
            scope="Original V726 capacity-benefit comparison only; a newly qualified source does not reverse this null.",
        ),
        arc_summary=dict(
            registry_reference=work["snapshots"]["ops/arc_solve_registry.yaml"],
            new_gameplay_executed=False,
            new_solve_count=0,
            frontier_inferred=False,
            claim_scope="Unchanged registry evidence only.",
        ),
        continuation_decisions=[
            dict(
                branch=r["branch"],
                decision="retain_authentic_source_then_qualify_runtime"
                if r["branch"] == "human_authority"
                else "blocked_require_exact_prerequisite",
                next_prerequisite=r["next_prerequisite"],
            )
            for r in table
        ]
        + [
            dict(
                branch="graph_collection",
                decision="deferred_unchanged",
                next_prerequisite="New historical-attempt topology authority before collection; retain original seven/ten bounds.",
            ),
            dict(
                branch="physical_hardware",
                decision="no_speed_or_joule_claim",
                next_prerequisite="New physical force-response implementation, parity, full service timing and measured energy; CPU susceptibility is unchanged.",
            ),
        ],
        retirement_recommendations=retirement(work),
    )
