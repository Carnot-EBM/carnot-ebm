"""REQ-REPORT-8957: bind execution authority before admitting new ARC outcomes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as base
from carnot.reporting import v765_branch_contract as prior
from carnot.reporting import v766_branch_evidence as evidence
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8957_v766_branch_contract",
    "exp8957-branch-contract",
    "2026.11.766",
)
PIN = "sha256:228b225dcb723a83e17aa83e6114f6bab61b3e3ae02efbe09ddbccf134c589c2"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/v766_branch_contract.py",
    "python/carnot/reporting/v766_branch_evidence.py",
    CLI,
]
TEST = "tests/python/test_v766_branch_contract_8957.py"
reference, checksum, seal, encode = prior.reference, prior.checksum, prior.seal, prior.encode
arc_support = evidence.arc_support


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the supervisor can distinguish work from a stall."""
    print(f"[exp8957] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: Any) -> Json:
    """Keep missing bytes, rejected authority and observed zero distinct."""
    return dict(
        upstream=TASK,
        **reference(path),
        check=check,
        artifact_field=field,
        expected=expected,
        observed=observed,
    )


def authority(root: Path) -> Json:
    """An independent digest prevents coordinated edits to all authority copies."""
    tasks, table, rows, failures = [], [], [], []
    mode, path = "unavailable", root / DESIGN
    try:
        text = path.read_text()
        table, tasks = parse_design(text, milestone=MILESTONE)
        active = yaml.load((root / ACTIVE).read_bytes(), Loader=yaml.CSafeLoader)
        staged = (
            yaml.load((root / STAGED).read_bytes(), Loader=yaml.CSafeLoader)
            if (root / STAGED).is_file()
            else None
        )
        selected = active if active["milestone"] == MILESTONE else staged
        if selected is None or selected["milestone"] != MILESTONE:
            raise ValueError("no_current_authority")
        mode = "activated" if active["milestone"] == MILESTONE else "staged"
        queues = [deepcopy(staged or selected), deepcopy(selected)]
        for queue in queues:
            Roadmap.model_validate(queue)
            for actual, expected in zip(queue["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(text, *queues, milestone=MILESTONE, first_id=8957, count=14)
        rows = checked["rows"]
        for i, row in enumerate(rows):
            row["checks"]["full_object"] = (
                i < min(len(tasks), *(len(q["tasks"]) for q in queues))
                and tasks[i] == queues[0]["tasks"][i] == queues[1]["tasks"][i]
            )
            row["matched"] = all(row["checks"].values())
        if not (
            checked["passed"]
            and all(r["matched"] for r in rows)
            and canonical_hash(tasks) == PIN
            and "<!-- V766_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("independent_authority_mismatch")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, yaml.YAMLError):
        failures.append(
            failure(
                path,
                "complete_authority",
                "full_task_objects",
                "14 exact ordered full objects, table, independent digest and queue",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    for i in range(14):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=f"exp{8957 + i}-missing",
                    order=i + 1,
                    checks=dict(independent_authority=False),
                    matched=False,
                )
            )
        row = rows[i]
        matched = row["matched"] and not failures
        row.update(
            matched=matched,
            value=int(matched),
            numerator=int(matched),
            denominator=1,
            absolute_metric=int(matched),
            cluster_id="administrative_contract",
            metric="ordered_task_contract_match",
            condition="exact_authority",
            arm="four_source_contract",
            censor_reason=None if matched else "authority_unavailable_or_rejected",
            effective_independent_groups=0,
            censored=not matched,
            excluded=False,
            seed=None,
            status="completed" if matched else "censored",
            missing_reason=None if matched else "authority_unavailable_or_rejected",
        )
    return dict(
        ready=not failures,
        rows=rows,
        tasks=tasks,
        table=table,
        failures=failures,
        comparison_mode=mode,
    )


def mutations(source: Path, raw: Path) -> list[Json]:
    """Change one private queue field at a time to prove causal rejection."""
    text = (source / DESIGN).read_text()
    name = STAGED if authority(source)["comparison_mode"] == "staged" else ACTIVE
    queue = yaml.load((source / name).read_bytes(), Loader=yaml.CSafeLoader)
    rows = []
    for mode in ["missing", "order", "prompt", "gate", "path"]:
        root, changed = raw / mode, deepcopy(queue)
        if mode == "missing":
            changed["tasks"].pop()
        if mode == "order":
            changed["tasks"].reverse()
        if mode == "prompt":
            changed["tasks"][0]["prompt"] += "changed"
        if mode == "gate":
            next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0]["artifact_field"] = (
                "forged"
            )
        if mode == "path":
            changed["tasks"][0]["deliverable"] = "results/forged.json"
        path = root / DESIGN
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        (root / ACTIVE).write_text(yaml.safe_dump(changed))
        rows.append(
            dict(
                mutation=mode,
                passed=not authority(root)["ready"],
                references=[reference(path), reference(root / ACTIVE)],
            )
        )
        progress("control_" + mode, len(rows), 5 - len(rows))
    return rows


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze operands before validation without starting models, devices or games."""
    progress("authenticate_before", 0, 14)
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    historical, arc = evidence.history(), evidence.arc_evidence()
    snapshots, failures = [], []
    for name in [DESIGN, ACTIVE, STAGED]:
        path = root / name
        if path.is_file():
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            snapshots.append(dict(seal(path, raw), current_execution_authority=True))
        elif name != STAGED:
            failures.append(
                failure(path, "authority_input", name, "original bytes", "missing_file")
            )
    snapshots.extend(dict(source=ref, **ref) for ref in historical["references"])
    for ref in arc["references"]:
        snapshots.append(seal(Path(ref["path"]), raw))
    for name in [
        *OWNED,
        TEST,
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/v755_branch_contract.py",
        "python/carnot/reporting/v755_branch_runner.py",
        "python/carnot/reporting/v765_branch_contract.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "python/carnot/reporting/arc_supervisor_frontier_8720.py",
        "python/carnot/agentic/arc_supervisor_refinement.py",
        "scripts/experiment_template.py",
        "ops/exclusion_manifest.yaml",
        "research-complete.yaml",
        "ops/conductor-log.md",
        "CLAUDE.md",
        "CODEX.md",
        "research-program.md",
        "ops/north-star.md",
        "openspec/capabilities/research-reporting/spec.md",
    ]:
        snapshots.append(dict(seal(ROOT / name, raw), executed_source=True))
    contract = authority(raw / "authority")
    controls = mutations(raw / "authority", raw / "controls") if contract["ready"] else []
    work = dict(
        contract=contract,
        history=historical,
        arc=arc,
        snapshots=snapshots,
        input_failures=failures,
        authority_mutation_rows=controls,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=dict(
            producer_argv=sys.orig_argv,
            repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
            free_bytes=shutil.disk_usage(raw).free,
            required_free_bytes=536870912,
            no_model_loads=True,
            no_device_probes=True,
            no_game_runs=True,
            activation_authorized=False,
            exclusion_manifest_edited=False,
            run_date_matches=run_date == "20261109",
            time_budget_s=4200,
            hard_cap_s=4800,
        ),
        raw_operands=[reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()],
    )
    atomic_json(raw / "work.json", work)
    progress(
        "authenticate_after",
        sum(r["matched"] for r in contract["rows"]),
        14 - sum(r["matched"] for r in contract["rows"]),
    )
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Administrative readiness survives absent ARC evidence and a quarantined old diagnostic."""
    end = max(
        [
            work["end_ns"],
            *[r.get("ended_monotonic_ns", work["end_ns"]) for r in validation["receipts"]],
        ]
    )
    prepared = dict(
        work,
        end_ns=end,
        manifests=[],
        methods=[],
        history=dict(work["history"], original_child_qualified=True),
        branches=dict(runtime=dict(ready=False), arc=dict(ready=False), failures=[]),
    )
    value = base.build(prepared, validation)
    tasks, controls = work["contract"]["tasks"], work["authority_mutation_rows"]
    producers = {t["id"]: t for t in tasks}
    gates = [
        dict(
            edge,
            required_field_declared=edge["artifact_field"]
            in _required_artifact_fields_block(producers[edge["upstream"]]["prompt"]),
        )
        for edge in base.edges(tasks)
    ]
    independent = all(
        not t.get("requires")
        and all(g["artifact_field"] != "contract_ready_score" for g in t["gated_on"])
        for t in tasks
    )
    checked = bool(
        validation["receipts"]
        and value["required_checks_passed"]
        and not any(r.get("timed_out", False) for r in validation["receipts"])
        and work["preconditions"]["run_date_matches"]
        and work["preconditions"]["repository_virtualenv"]
        and work["preconditions"]["free_bytes"] >= work["preconditions"]["required_free_bytes"]
        and value["duration_s"] < 4200
        and all(r["passed"] for r in controls)
    )
    ready = bool(
        checked
        and work["contract"]["ready"]
        and not work["input_failures"]
        and len(controls) == 5
        and independent
        and all(r["required_field_declared"] for r in gates)
    )
    arc_ready = int(checked and work["arc"]["ready"])
    failures = (
        work["contract"]["failures"]
        + work["input_failures"]
        + work["history"]["failures"]
        + work["arc"]["failures"]
    )
    verdict = "disqualified" if not checked else "blocked" if failures or not ready else "null"
    for field in [
        "preserved_v754_reference",
        "runtime_repair_ready_score",
        "arc_new_outcomes_available_score",
        "branch_admission_rows",
    ]:
        value.pop(field)
    value.update(
        experiment_id=8957,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7668957,
        honest_verdict=f"complete_{verdict}_v766_branch_contract",
        verdict_class=verdict,
        contract_ready_score=int(ready),
        arc_new_outcomes_ready_score=arc_ready,
        required_checks_passed=checked,
        model_specs=[],
        execution_venue="host",
        intended_count=14,
        censored_count=14 - value["completed_count"],
        preconditions_checked=work["preconditions"],
        full_task_digest=canonical_hash(tasks),
        canonical_tasks_sha256=PIN,
        prior_scope_rows=[dict(task_id=t["id"], **p) for t in tasks for p in t["prior_failures"]],
        gate_edge_rows=gates,
        authority_mutation_rows=controls,
        history=work["history"],
        authority_comparison_mode=work["contract"]["comparison_mode"],
        execution_snapshot=[s for s in work["snapshots"] if s.get("current_execution_authority")],
        arc_frontier=work["arc"],
        arc_support_rows=work["arc"]["support"]["support_rows"],
        gate_check_summary=failures,
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            owned_validation=checked,
            independent_science_roots=independent,
            exact_gate_fields=all(r["required_field_declared"] for r in gates),
            mutation_controls=len(controls) == 5 and all(r["passed"] for r in controls),
            graph_corpus_authenticated=work["history"]["authenticated"],
            new_arc_support=bool(arc_ready),
        ),
        acceptance_gate_principles=dict(
            contract="Full task equality prevents disappearing tasks.",
            history="Quarantine does not erase independent graph evidence.",
            arc="Only new authenticated uncensored held-family evidence permits analysis.",
        ),
        code_config_hashes=[s for s in work["snapshots"] if s.get("executed_source")],
        reproduction_commands=[
            f"PYTHONPATH=python .venv/bin/python -u {CLI} --date {work['run_date']}",
            f".venv/bin/python -u {CLI} --cold-replay <primary-path>",
        ],
    )
    value["raw_shard_hashes"] += work["raw_operands"]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Exact input bytes, task identity and measured owned checks bind administrative readiness. Historical model calls remain retained evidence; current calls are zero. ARC counts unique newly authenticated uncensored episodes by family and both arms. Graph evidence does not establish the oracle-distinct north star.",
    )
    principles = [
        (
            [
                "experiment_id",
                "task_id",
                "milestone",
                "run_date",
                "random_seed",
                "reproducibility_checksum",
            ],
            "Bind the scheduled task, input bytes and reproducible randomness.",
        ),
        (
            ["honest_verdict"],
            "An honest terminal outcome avoids wasted retries and preserves exact rerun retirement.",
        ),
        (
            ["verdict_class"],
            "The closed class carries claim strength; an oracle verifier or failed acceptance gate forbids positive.",
        ),
        (
            ["inference_substrate", "inference_substrate_class", "execution_venue"],
            "Declare the actual legal substrate and venue. No model load claims no new GGUF execution.",
        ),
        (
            ["MODEL_SPECS", "model_specs", "model_invocation_counts", "numerical_training_calls"],
            "Count current generator calls, small-head training and retained outputs separately.",
        ),
        (
            ["duration_s", "phase_spans", "preconditions_checked"],
            "Actual clocks and resource receipts prevent fabricated execution and duration padding.",
        ),
        (
            ["verifier_is_oracle", "exposure_scope", "independent_generalization_score"],
            "Exact graph labels do not establish the oracle-distinct north star.",
        ),
        (
            [
                "intended_count",
                "completed_count",
                "failed_count",
                "censored_count",
                "excluded_count",
                "independent_count",
                "rows",
            ],
            "Retain every scheduled unit; typed rows recompute metrics without inflating independent sample size.",
        ),
        (
            ["gate_check_summary"],
            "Name upstream, content hash, exact field, expected and observed values; distinguish absence, zero and rejection.",
        ),
        (
            [
                "source_artifact_hashes",
                "code_config_hashes",
                "raw_shard_hashes",
                "field_principles",
            ],
            "Retain content custody and the reasons each field exists.",
        ),
        (
            [
                "acceptance_gates",
                "required_checks_passed",
                "owned_statement_coverage",
                "validation_receipts",
                "global_test_debt",
            ],
            "Owned checks must pass; global debt stays separate from administrative readiness and scientific benefit.",
        ),
        (
            ["contract_ready_score"],
            "Full task equality prevents a promised task from disappearing.",
        ),
        (
            ["arc_new_outcomes_ready_score"],
            "Skip costly ARC analysis when new supported live outcomes do not exist.",
        ),
        (
            [
                "full_task_objects",
                "full_task_digest",
                "visible_task_table",
                "gate_edge_rows",
                "prior_scope_rows",
                "arc_frontier",
                "arc_support_rows",
            ],
            "Freeze complete execution authority and the exact evidence frontier before downstream reads.",
        ),
    ]
    value["field_principles"].update(
        {field: principle for fields, principle in principles for field in fields}
    )
    value["acceptance_gate_principles"] = {
        gate: value["field_principles"]["acceptance_gates"] for gate in value["acceptance_gates"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Recompute sealed authority and support so fresh aggregate hashes cannot invent evidence."""
    try:
        value = json.loads(path.read_bytes())
        if not value["required_checks_passed"] or not verify_measured(
            value["owned_statement_coverage"], OWNED
        ):
            raise ValueError("unqualified_statements")
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        ref = value["raw_shard_hashes"][0]
        work = json.loads(Path(ref["path"]).read_bytes())
        if work["contract"] != authority(Path(work["raw"]) / "authority"):
            raise ValueError("authority_drift")
        for control in work["authority_mutation_rows"]:
            if control["passed"] is not (
                not authority(Path(work["raw"]) / "controls" / control["mutation"])["ready"]
            ):
                raise ValueError("mutation_control_drift")
        if work["history"] != evidence.history():
            raise ValueError("historical_drift")
        ledger = (
            next(
                s
                for s in work["snapshots"]
                if s["source"]["path"] == str(ROOT.joinpath(*evidence.arc.DEFAULT_LEDGER_PARTS))
            )
            if work["arc"]["references"]
            and any(
                s["source"]["path"] == str(ROOT.joinpath(*evidence.arc.DEFAULT_LEDGER_PARTS))
                for s in work["snapshots"]
            )
            else None
        )
        replayed_arc = evidence.arc_evidence(ledger_path=Path(ledger["path"]) if ledger else None)
        if ledger:
            replayed_arc["references"] = [
                dict(r, path=ledger["source"]["path"]) if r["path"] == ledger["path"] else r
                for r in replayed_arc["references"]
            ]
            for row in replayed_arc["failures"]:
                if row["path"] == ledger["path"]:
                    row["path"] = ledger["source"]["path"]
        if work["arc"] != replayed_arc:
            raise ValueError("arc_frontier_drift")
        validation_path = Path(work["raw"]) / "validation.json"
        validation = (
            json.loads(validation_path.read_bytes())
            if validation_path.is_file()
            else dict(
                receipts=value["validation_receipts"],
                coverage=value["owned_statement_coverage"],
                global_test_debt=value["global_test_debt"],
            )
        )
        expected = build(dict(work, raw_reference=ref), validation)
        if validation_path.is_file():
            expected["raw_shard_hashes"].append(reference(validation_path))
        expected["reproducibility_checksum"] = checksum(expected)
        passed = value == expected and checksum(value) == value["reproducibility_checksum"]
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return dict(passed=False, reason="invalid_or_changed_evidence")
