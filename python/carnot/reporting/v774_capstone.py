"""REQ-REPORT-9081-V774: complete accounting preserves independent scientific limits."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import scoped_provenance_9014 as runner
from carnot.reporting import v774_dispatch_contract as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.encoder_head_diagnosis_8972 import checksum, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.invocation_provenance_9000 import environment
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v764_capstone_sources import retain
from carnot.reporting.v766_branch_evidence import arc_support
from carnot.reporting.v768_capstone import edge

Json = dict[str, Any]
ROOT, MILESTONE = authority.ROOT, authority.MILESTONE
NAME, TASK = "experiment_9081_v774_capstone", "exp9081-capstone"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v774_capstone_9081.py"
OWNED = ["python/carnot/reporting/v774_capstone.py", CLI]
DIRECT = [
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_conductor_gates.py",
]
REUSED = list(
    dict.fromkeys(
        [
            *runner.REUSED,
            *authority.OWNED,
            "python/carnot/reporting/v768_capstone.py",
            "python/carnot/reporting/v764_capstone_sources.py",
            "python/carnot/reporting/v766_branch_evidence.py",
            "scripts/publication_gate.py",
        ]
    )
)
MODEL_SPECS: list[Json] = []
PINS = {
    "research-roadmap.yaml": "sha256:1d87c5c7f1e4ab81c81333f67c6266d1e78f290ba0b28abc130d93c5326847bd",
    "ops/conductor-log.md": "sha256:884424c737c3c18bca7f960683d83458fa7285b81013491db1d97d7cbb828442",
    "ops/exclusion_manifest.yaml": "sha256:081d0cb3f30a97c2afae61715d9c84de5254acae75eef050e31a3a221bdfc8a0",
    "results/experiment_9030_v771_retained_shard_a.json": "sha256:9755b2dd55459cc48361deb85cbc2bae16e6fc80f231373d56d7033486033944",
    "results/experiment_9046_fresh_corpus_b.json": "sha256:da6a56e3bc89675c59c7ef8857fb563203f6e9c12689c695f2ad31d9f965d8bf",
    "results/experiment_9056_v773_human_release_authority.json": "sha256:5a0f0e0fd0831b61fd30f03a311c50327f46c6676f4a77fe33b431eacc86224a",
    "results/experiment_9057_human_probe_methods.json": "sha256:ec2fb0d9952cf7e48db583e539879164bd15b81c4904d43a05f637be703b13c8",
    "results/experiment_9060_sealed_corpus_admission.json": "sha256:d920be68911ea7ce262f81361e9c3bdde0ea0767ca54debc165140686111915f",
    "results/experiment_9066_arc_transfer_evidence.json": "sha256:55ad80aaf449877fffc5c3ac38dca78b44561b6368104592dbf63b4d61c45c50",
    "results/experiment_9067_v773_capstone.json": "sha256:eb76ada3c0777398b8ac9c6910a59678400ab3b7deb7f8eb92048412c14bcf75",
    "results/experiment_9068_v774_dispatch_contract.json": "sha256:fdb97d887079941cfa17e89146cf6097deb2396d2007a44565020a755f10f090",
    "results/experiment_9069_v774_source_custody.json": "sha256:21a8e31edb572b34ea638fb78591b993e8c40e0855bfa4695ae3cbd9c4c295ee",
    "results/experiment_9070_v774_adjudication_methods.json": "sha256:aee3a477ea80a3ec5b0d7187b1f3f0b92001c34e1fb5a732a470bdfd2ffe8e73",
    "results/experiment_9071_collect_b.json": "sha256:3e3c2bec3c2257da5ff31619ba38fa1eb2e5899d07cc700343ba408dd3f6f315",
    "results/experiment_9073_human_features.json": "sha256:b080dafca3ff8093a4acf92c4ad636f3ab76dc043a736bd63d0e51c04325d109",
    "results/experiment_9079_v774_hardware_boundary.json": "sha256:2e3c1a9581e1f498f40d105efacd45a41b04b9aa03b75297c5bce764bdfbba8f",
    "results/experiment_9080_arc_transfer.json": "sha256:6a5794fc9227b048c2453d9aaed2e2eae1c61bf2e8885cf7b6e7ae7150da7898",
    "openspec/change-proposals/research-roadmap-vNEXT.md": "sha256:9737d705002f952c03fc9f0a57e3404e8ad8efc5a7119af10d891d967fad8b10",
}
PINS.update(
    {
        "AGENTS.md": "sha256:6afad92d58a6c7860487402caf9b70c6c9a0a8734724efc8112fd200e1c8a93a",
        "CODEX.md": "sha256:227bf72d8d45fbf9da3af2ff7ed6e176400e4c25c0282de2780c587b377da3dc",
        "CLAUDE.md": "sha256:5c77c0d69019f3767a2e842f2808a5f599a3520ae7c59256266e21dc1573e1b5",
        "research-program.md": "sha256:53ed14d8257e92794b841f3959dc3337cafe56adce714fe3447fd62977e4d9f9",
        "ops/north-star.md": "sha256:ade6a23f936f2a0cf8e2da037cd1c34a468c0f0c25dad15dc37209b0d2323eb0",
        "ops/e2e-test-plan.md": "sha256:47b7112b8307604c983afb90c624fdd34441b982f19ab9e9839e657c386c7646",
        "research-references.md": "sha256:a3d9006a7dac3f09f469a14d39541a0f60436eb4c7aedca7fe9c3db599e6a903",
        "openspec/change-proposals/research-roadmap-v773-preserved-20261112.md": "sha256:96163ef0782d8c6125561eec52e89d4d8982a8829bc0dd41704a9ee5ecac57c4",
        "python/carnot/reporting/v773_capstone.py": "sha256:edb8ed2b4691ba08089a2bddabdfb5796e9cbabde262fbce6f9b9b8c149dba4b",
        "python/carnot/reporting/roadmap_contract.py": "sha256:eed75373b8b00f3dc6442fbf878cc833ec7bcf44f6c258019eebcb3e89e4ea63",
        "scripts/experiment_template.py": "sha256:9396db4d8604620a2dc0f26b7b797099e5215d2f1d681cd01a7d1ed7c423c184",
        "scripts/conductor_gates.py": "sha256:d406171cde38a594a78e7bff8958212e722fced9fc583b2b1858f8fc9d3732ca",
    }
)
HEALTH = "results/raw/" + NAME + "/development/global_python_suite.receipt.json"

DEPENDENCIES = {
    "pytest": "9.0.3",
    "coverage": "7.14.1",
    "ruff": "0.15.17",
    "mypy": "2.2.0",
    "PyYAML": "6.0.3",
    "numpy": "2.5.1",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts while existing child supervision supplies heartbeats."""
    print(f"[exp9081] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Keep original validators and consumers while freezing this capstone's scope."""
    with patch.multiple(
        runner,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        DIRECT=DIRECT,
        REUSED=REUSED,
        progress=progress,
    ):
        yield


def collect(raw: Path, run_date: str, root: Path = ROOT) -> Json:
    """Authenticate all required bytes before accounting; failed resources stop reduction."""
    progress("preconditions_before")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="carnot-9081-"))
    with binding():
        manifest = runner.scope_manifest(private)
    env = environment(raw)
    checks = list(env["checks"])
    checks += [
        authority.check(raw, "run_date", "20261113", run_date, "date"),
        authority.check(raw, "private_mode", 0o700, raw.stat().st_mode & 0o777, "storage"),
    ]
    checks += [
        authority.check(
            ROOT / "pyproject.toml", name, version, importlib.metadata.version(name), "dependency"
        )
        for name, version in DEPENDENCIES.items()
    ]
    snapshots: Json = {}
    for index, (name, pin) in enumerate(PINS.items()):
        path = root / name
        checks.append(
            authority.check(
                path, "sha256", pin, reference(path)["sha256"] or "missing_file", "source_bytes"
            )
        )
        if checks[-1]["passed"]:
            snapshots[name] = retain(path, raw, pin)
        progress("source_authentication", index + 1, len(PINS) - index - 1)
    code = [retain(Path(r["path"]), raw, r["sha256"]) for r in manifest["code_test_hashes"]]
    work: Json = dict(
        raw=str(raw),
        private=str(private),
        run_date=run_date,
        started_ns=started,
        frozen_ns=time.monotonic_ns(),
        validation_scope_manifest=manifest,
        preconditions_checked=dict(env, checks=checks),
        snapshots=snapshots,
        references=list(snapshots.values()),
        code=code,
        values={},
        tasks=[],
        reduction={},
        validation=dict(receipts=[], coverage=dict(complete=False)),
        publication_receipts=[],
    )
    if (root / HEALTH).is_file():
        ref = retain(root / HEALTH, raw, reference(root / HEALTH)["sha256"])
        work["current_suite"] = json.loads(Path(ref["path"]).read_bytes())
        work["references"].append(ref)
        for log_ref in work["current_suite"]["stream_references"]:
            work["references"].append(retain(Path(log_ref["path"]), raw, log_ref["sha256"]))
    if all(r["passed"] for r in checks):
        tasks = yaml.safe_load(Path(snapshots["research-roadmap.yaml"]["path"]).read_bytes())[
            "tasks"
        ]
        require(canonical_hash(tasks) == authority.PIN, "task_authority")
        work["tasks"] = tasks
        for task in tasks[:-1]:
            n = int(task["id"][3:7])
            names = [s for s in snapshots if s.startswith(f"results/experiment_{n}_")]
            if names:
                ref = snapshots[names[0]]
                value = json.loads(Path(ref["path"]).read_bytes())
                work["values"][task["id"]] = dict(
                    value=value,
                    reference=ref,
                    selected_path=str(root / names[0]),
                    selected_name=names[0],
                )
        operands = {
            r["sha256"]: r
            for e in work["values"].values()
            for r in e["value"].get("raw_shard_hashes", [])
        }
        for index, ref in enumerate(operands.values()):
            path = Path(ref["path"])
            checks.append(
                authority.check(
                    path,
                    "sha256",
                    ref["sha256"],
                    reference(path)["sha256"] or "missing_file",
                    "primitive_bytes",
                )
            )
            if checks[-1]["passed"]:
                work["references"].append(retain(path, raw, ref["sha256"]))
            if index % 50 == 0:
                progress("primitive_authentication", index + 1, len(operands) - index - 1)
        if all(r["passed"] for r in checks):
            progress("publication_gate_before")
            receipt = runner.run_check(
                CommandSpec(
                    "publication_gate",
                    (str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"),
                    "historical_FoVer",
                    60,
                ),
                raw / "publication",
                manifest,
            )
            checks.append(
                authority.check(
                    Path(receipt["log_path"]),
                    "normal_exit",
                    True,
                    receipt["passed"] and receipt["normal_exit"],
                    "publication_gate_execution",
                )
            )
            work["publication_receipts"] = [receipt]
            if receipt["passed"]:
                work["publication_gate"] = json.loads(Path(receipt["log_path"]).read_bytes())
            progress("publication_gate_after")
    progress("preconditions_after", len(checks), sum(not r["passed"] for r in checks))
    if all(r["passed"] for r in checks):
        progress("reduction_before")
        work["reduction"] = reduce(work)
        progress("reduction_after", 14, 0)
    work["ended_ns"] = time.monotonic_ns()
    atomic_json(raw / "work.json", work)
    return work


def slots(work: Json) -> list[Json]:
    """Scheduler skips and title-only completion cannot supply a scientific verdict."""
    log = Path(work["snapshots"]["ops/conductor-log.md"]["path"]).read_text().splitlines()
    rows = []
    for task in work["tasks"][:-1]:
        entry = work["values"].get(task["id"], {})
        value = entry.get("value", {})
        events = [line for line in log if task["title"][:50].strip() in line]
        scheduler = value.get("blocked_at_layer") == "conductor_pre_gate"
        qualified = bool(
            value.get("required_checks_passed") and not value.get("flagged_adversarial")
        )
        drift = [
            dict(r, observed=reference(Path(r["path"]))["sha256"])
            for r in value.get("code_config_hashes", [])
            if reference(Path(r["path"]))["sha256"] != r["sha256"]
        ]
        rows.append(
            dict(
                task_id=task["id"],
                experiment_id=int(task["id"][3:7]),
                task=task,
                accounted=True,
                qualified=qualified and not drift,
                primary_path=task["deliverable"],
                selected_path=entry.get("selected_path"),
                primary_status="present"
                if entry.get("selected_path") == str(ROOT / task["deliverable"])
                else "missing_file",
                reference=entry.get("reference"),
                schema=value.get("schema"),
                disposition="authenticated_scheduler_skip"
                if scheduler
                else "producer_primary"
                if entry
                else "title_only_OK"
                if any("| OK |" in e for e in events)
                else "missing_primary",
                honest_verdict=None if scheduler else value.get("honest_verdict"),
                verdict_class=None if scheduler else value.get("verdict_class"),
                current_evidence_verdict_class="disqualified"
                if drift
                else value.get("verdict_class"),
                scheduler_honest_verdict=value.get("honest_verdict") if scheduler else None,
                scheduler_events=events,
                retired_upstream_observed=any("upstream_retired" in e for e in events),
                current_code_drift=drift,
            )
        )
    return rows


def reduce(work: Json) -> Json:
    """Keep unavailable predictions unmeasured and independently recount available frontier rows."""
    require(canonical_hash(work["tasks"]) == authority.PIN, "task_authority")
    values = work["values"]
    # shortcut: no qualified prediction primaries exist; add their registered reducer when they do.
    require(
        not any(
            values.get(t, {}).get("value", {}).get("required_checks_passed")
            for t in ["exp9076-graph-benefit", "exp9077-human-sensitivity", "exp9078-service-cost"]
        ),
        "qualified_raw_reducer_required",
    )
    branch = values["exp9068-dispatch-contract"]["value"]
    human = values["exp9070-adjudication-methods"]["value"]
    result: Json = {}
    for name, contract in [
        ("seven_bound_rows", "seven_bound_contract"),
        ("ten_bound_rows", "ten_bound_contract"),
    ]:
        policy = branch["frozen_science"][contract]
        result[contract] = policy
        result[name] = [
            dict(
                bound_id=f"{name}:{i + 1}",
                **bound,
                value=None,
                numerator=None,
                denominator=0,
                status="unmeasured",
                observed="missing_file",
                passed=False,
                upstream="exp9076-graph-benefit",
                path=work["tasks"][8]["deliverable"],
            )
            for i, bound in enumerate(policy["bounds"])
        ]
    policy = human["human_bound_contract"]
    result["human_bound_contract"] = policy
    result["human_bound_rows"] = [
        dict(
            bound_id=f"human:{i + 1}",
            control=control,
            criterion="lower>0",
            metric=policy["metric"],
            bootstrap_unit=policy["bootstrap_unit"],
            tail_probability=policy["one_sided_tail"],
            original_label_mapping=human["original_label_mapping"],
            target_panel=human["target_panel"],
            value=None,
            numerator=None,
            denominator=0,
            status="unmeasured",
            observed="missing_file",
            passed=False,
            upstream="exp9077-human-sensitivity",
            path=work["tasks"][9]["deliverable"],
        )
        for i, control in enumerate(policy["required_improvements"])
    ]
    source_rows = values["exp9069-source-custody"]["value"]["rows"]
    require(len({r["unit_id"] for r in source_rows}) == len(source_rows), "source_identity")
    result["source_completion_rows"] = [
        dict(
            arm=arm,
            intended_count=sum(r["arm"] == arm for r in source_rows),
            qualified_count=sum(r["arm"] == arm and bool(r["value"]) for r in source_rows),
            retained_transport_count=sum(
                r["arm"] == arm and r.get("disposition") == "retained_transport"
                for r in source_rows
            ),
            B_qualified_artifact=False,
            original_custody_verdict=values["exp9069-source-custody"]["value"]["honest_verdict"],
            primitive_rows_reference=values["exp9069-source-custody"]["reference"],
        )
        for arm in ["A", "B"]
    ]
    result["cost_quality_summary"] = dict(
        status="unmeasured",
        observed="missing_file",
        cost_ratio_upper95=None,
        complete_cost_recommendation=False,
        deployment_ready=False,
        path=work["tasks"][10]["deliverable"],
    )
    hardware = values["exp9079-hardware-boundary"]["value"]
    result["hardware_summary"] = {
        k: hardware[k]
        for k in [
            "honest_verdict",
            "per_board_rows",
            "operation_support_rows",
            "acceleration_bound",
            "nonaccelerable_fraction",
            "procurement_conditions",
            "physical_acceleration_demonstrated",
        ]
    }
    ref = next(
        r for r in work["references"] if r["sha256"] == branch["raw_shard_hashes"][0]["sha256"]
    )
    primitive = json.loads(Path(ref["path"]).read_bytes())["reduction"]["arc"]
    support = arc_support(primitive["new_event_rows"])
    require(
        support == primitive["support"] == branch["arc_frontier"]["support"],
        "arc_primitive_support",
    )
    result["arc_summary"] = dict(
        primitive, **support, source_reference=ref, new_gameplay_executed=False
    )
    result["title_key_rows"] = [
        dict(
            unit_id=t["id"],
            dispatch_key=t["title"][:50].strip(),
            passed=sum(t["title"][:50].strip() == u["title"][:50].strip() for u in work["tasks"])
            == 1,
        )
        for t in work["tasks"]
    ]
    result["continuation_evidence"] = [
        dict(scope=s, status=status, reopening_evidence=e)
        for s, status, e in [
            (
                "MDH-Bench",
                "closed_unchanged",
                "New original individual judgments and original factuality criterion; do not substitute the new panel.",
            ),
            (
                "MDHJudgments",
                "blocked_authority",
                "Author-authenticated base-question families and complete exposure closure; insufficient measured support cannot be repaired by relaxing the cohort.",
            ),
            (
                "frozen_graph_comparison",
                "blocked_unmeasured",
                "Qualified paired prediction seals. A fully measured null favors converged linear and ends this frozen spline comparison. New sampling needs a distinct hypothesis, never favorable-cohort hunting.",
            ),
            (
                "cost_and_hardware",
                "blocked_unmeasured",
                "Complete paired service quality, timing, recovery and demand evidence before recommendation.",
            ),
            (
                "ARC_transfer",
                "blocked_no_new_support",
                "At least30 new uncensored episodes, three held families and both arms within each; public replay supplies no hidden-game evidence.",
            ),
        ]
    ]
    return result


def build(work: Json) -> Json:
    """Qualified administrative accounting can complete while independent science stays blocked."""
    failed = [r for r in work["preconditions_checked"]["checks"] if not r["passed"]]
    science = (
        reduce(work)
        if not failed
        else dict(
            seven_bound_rows=[],
            ten_bound_rows=[],
            human_bound_rows=[],
            source_completion_rows=[],
            hardware_summary={},
            arc_summary={},
            cost_quality_summary={},
            title_key_rows=[],
            continuation_evidence=[],
        )
    )
    with binding():
        owned = bool(
            not failed
            and runner.receipts_pass(work)
            and verify_measured(work["validation"]["coverage"], OWNED)
            and not work["validation"].get("terminal_failed", False)
        )
    cls = "blocked" if failed or owned else "disqualified"
    reason = failed[0]["check"] if failed else "declared_gate" if owned else "owned_validation"
    verdict = f"complete_{cls}_{reason}"
    outcomes = slots(work) if not failed else []
    outcomes.append(
        dict(
            task_id=TASK,
            experiment_id=9081,
            accounted=not failed,
            qualified=owned,
            disposition="self_slot",
            honest_verdict=verdict,
            verdict_class=cls,
        )
    )
    retirement = []
    for task, slot in zip(work["tasks"], outcomes, strict=not bool(failed)):
        for prior in task["prior_failures"]:
            refs = [
                r
                for name, r in work["snapshots"].items()
                if name.startswith(f"results/experiment_{prior['experiment_id'][3:7]}_")
            ]
            original = json.loads(Path(refs[0]["path"]).read_bytes()) if refs else {}
            actual = slot.get("honest_verdict") or slot.get("scheduler_honest_verdict")
            previous = original.get("honest_verdict")
            retirement.append(
                dict(
                    task_id=task["id"],
                    prior_experiment_id=prior["experiment_id"],
                    planned_prior_verdict=prior["verdict"],
                    actual_prior_verdict=previous,
                    prior_verdict_source=prior.get("verdict_source", "original_primary"),
                    actual_current_verdict=actual,
                    current_verdict_source=slot["disposition"],
                    changed_mechanism=prior["addressed_by"],
                    reopening_evidence=prior["addressed_by"],
                    prior_reference=refs[0] if refs else None,
                    retire_if_same_verdict=True,
                    unchanged_verdict=previous is not None and previous == actual,
                    outcome="retire_unchanged_scope"
                    if previous is not None and previous == actual
                    else "preserve_original_require_changed_evidence",
                    manifest_mutated=False,
                )
            )
    gates = [edge(g, work["values"]) for task in work["tasks"] for g in task["gated_on"]]
    upstream_checks = [
        r
        for e in work["values"].values()
        for r in e["value"].get("gate_check_summary", [])
        if isinstance(r, dict)
    ]
    publication = work.get(
        "publication_gate",
        dict(
            gates=dict.fromkeys(["G1", "G2", "G3", "G4"], None),
            paper_ready=False,
            unmet_gates=["not_executed"],
        ),
    )
    elapsed = (work["ended_ns"] - work["started_ns"]) / 1e9
    branch = work["values"].get("exp9068-dispatch-contract", {}).get("value", {})
    historical = branch.get("repository_health", {})
    value: Json = dict(
        experiment_id=9081,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=work["run_date"],
        honest_verdict=verdict,
        verdict_class=cls,
        verifier_is_oracle=True,
        independent_generalization_score=0,
        claim_scope="Fourteen administrative outcomes. No measured graph, selected human-panel, broad no-cheap-oracle or complete-cost benefit. Graph wins remain circular_positive; human agreement is neither medical correctness nor general verifier benefit.",
        exposure_scope="Original metadata and retained historical outcomes. Protected TEST truths and human targets remain sealed; source-access and family-exposure authority remain unresolved.",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        intended_count=14,
        completed_count=14 if not failed else 0,
        failed_count=0,
        censored_count=14 if failed else 0,
        excluded_count=0,
        independent_count=0,
        rows=[
            dict(
                unit_id=f"exp{9068 + i}",
                group_id="administrative_contract",
                arm="reconciliation",
                condition="task_accounting",
                status="censored" if failed else "completed",
                metric="task_slot_accounted",
                value=0 if failed else 1,
                numerator=0 if failed else 1,
                denominator=1,
                exclusion_reason=None,
            )
            for i in range(14)
        ],
        reconciliation_ready_score=int(owned),
        slot_rows=outcomes,
        full_task_objects=work["tasks"],
        full_task_digest=canonical_hash(work["tasks"]),
        **science,
        retirement_scope_rows=retirement,
        gate_check_summary=[*work["preconditions_checked"]["checks"], *upstream_checks, *gates],
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        validation_scope_manifest=work["validation_scope_manifest"],
        validation_receipts=[
            *work["publication_receipts"],
            *work["validation"]["receipts"],
            *work["validation"].get("terminal_receipts", []),
        ],
        owned_statement_coverage=work["validation"]["coverage"],
        repository_health=dict(
            historical=historical,
            current_suite=work.get("current_suite", {}),
            affects_owned_readiness=False,
        ),
        repository_health_passed=False,
        preconditions_checked=work["preconditions_checked"],
        duration_s=elapsed,
        random_seed=774081,
        phase_spans=work.get(
            "phase_spans", [dict(phase="authentication_and_accounting", duration_s=elapsed)]
        ),
        raw_shard_hashes=work["references"],
        code_config_hashes=work["code"],
        work_reference=reference(Path(work["raw"]) / "work.json"),
        cited_upstream_artifacts=[
            dict(
                e["reference"],
                experiment_id=int(task[3:7]),
                fields_imported=["original_outcome", "gates", "row_and_seal_references"],
                historical_model_calls_not_imported=True,
            )
            for task, e in work["values"].items()
        ],
        publication_scope="historical_FoVer",
        historical_publication_gate=publication,
        **{k: publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        current_science_paper_ready=False,
        activation_authorized=False,
        external_publication_authorized=False,
        methodology_note="Authenticate exact sources before reduction. Recount primitive custody rows and ARC delta; unavailable paired predictions yield unmeasured bounds. Current owned checks qualify accounting only. Historical model work and repository health remain separate.",
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Immutable operands and primitive denominators bind accounting; complete external blocking grants no scientific or publication claim.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild summaries from independently pinned sources; rehashing cannot create a win."""
    try:
        value = json.loads(path.read_bytes())
        require(checksum(value) == value["reproducibility_checksum"], "checksum_drift")
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        require(
            reference(Path(value["work_reference"]["path"])) == value["work_reference"],
            "work_drift",
        )
        for name, ref in work["snapshots"].items():
            require(ref["sha256"] == PINS[name], "original_pin")
        for ref in work["references"] + work["code"]:
            require(reference(Path(ref["path"]))["sha256"] == ref["sha256"], "operand_drift")
        for entry in work["values"].values():
            require(
                entry["reference"] == work["snapshots"][entry["selected_name"]]
                and entry["value"] == json.loads(Path(entry["reference"]["path"]).read_bytes()),
                "producer_drift",
            )
        if work["publication_receipts"]:
            receipt = work["publication_receipts"][0]
            require(
                reference(Path(receipt["log_path"]))["sha256"] == receipt["log_sha256"]
                and work["publication_gate"] == json.loads(Path(receipt["log_path"]).read_bytes()),
                "publication_gate_drift",
            )
        require(value == build(work), "reduction_drift")
        return dict(passed=True, replay_passed=True, slots=14)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        return dict(passed=False, replay_passed=False, reason=str(error))


def validate(work: Json) -> Json:
    """Run the frozen scope once and reopen the actual statement measurements."""
    private, raw = Path(work["private"]), Path(work["raw"])
    with binding(), patch.dict(os.environ, COVERAGE_RCFILE=str(private / "coverage.ini")):
        plan = runner.validation_plan(private)
        require(
            [dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in plan]
            == work["validation_scope_manifest"]["commands"],
            "frozen_commands",
        )
        receipts = [
            runner.run_check(c, raw / "validation", work["validation_scope_manifest"]) for c in plan
        ]
    report = measured([private / ".coverage"], OWNED, raw / "coverage")
    report.pop("rows", None)
    return dict(receipts=receipts, coverage=report)


def main(argv: list[str] | None = None) -> int:
    """Publish one checked primary; private replay never changes original determinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261113"], default="20261113")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.from_work and args.output.absolute().is_relative_to(ROOT):
        parser.error("private replay publication requires output outside the checkout")
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    raw = args.output.absolute().parent / "raw" / NAME / "invocations" / str(time.time_ns())
    try:
        if args.from_work:
            work = json.loads(args.from_work.read_bytes())
            work["raw"] = str(raw)
        else:
            work = collect(raw, args.date)
            require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 3300, "measurement_deadline")
            if all(r["passed"] for r in work["preconditions_checked"]["checks"]):
                progress("owned_validation_before")
                work["validation"] = validate(work)
                progress("owned_validation_after")
        work["phase_spans"] = [
            dict(
                phase="authentication_and_accounting",
                duration_s=(work["ended_ns"] - work["started_ns"]) / 1e9,
            ),
            dict(phase="validation", duration_s=(time.monotonic_ns() - work["ended_ns"]) / 1e9),
        ]
        work["ended_ns"] = time.monotonic_ns()
        atomic_json(raw / "work.json", work)
        value = build(work)
        atomic_json(raw / "candidate.json", value)
        require(
            value["required_checks_passed"]
            or not all(r["passed"] for r in work["preconditions_checked"]["checks"]),
            "owned_validation",
        )
        with binding():
            report = runner.terminal(
                raw / "candidate.json", raw / "preflight", work["validation_scope_manifest"]
            )
            work["validation"].update(
                terminal_failed=not report["passed"], terminal_receipts=report["receipts"]
            )
            atomic_json(raw / "work.json", work)
            value = build(work)
            require(report["passed"], "terminal_checks")
            require((time.monotonic_ns() - work["started_ns"]) / 1e9 < 4800, "hard_deadline")
            published = publish_primary(
                args.output,
                value,
                lambda p: runner.terminal(
                    p, raw / "publication_checks", work["validation_scope_manifest"]
                ),
            )
        progress("complete", 14, 0)
        print(json.dumps(published), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps(dict(passed=False, reason=str(error))), flush=True)
        return 1
