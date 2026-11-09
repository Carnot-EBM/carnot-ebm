"""REQ-REPORT-8944: exact task custody is administrative readiness, never detector benefit."""

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
from carnot.reporting import v764_branch_contract as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v749_contract_methods import encode as encode
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8944_v765_branch_contract",
    "exp8944-branch-contract",
    "2026.11.765",
)
PIN = "sha256:05f7082c6ee1cbab43dbe83a197646209d42a449d0f1c2c1907d753e61a21115"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v765_branch_contract.py", CLI]
TEST = "tests/python/test_v765_branch_contract_8944.py"
HISTORY = "results/experiment_8943_v764_capstone.json"
LITERATURE = "research-references.md"
ARCHIVE = "openspec/change-proposals/research-roadmap-v764-preserved-20261109.md"
QUEUE = "openspec/change-proposals/research-roadmap-v764-queue-preserved-20261109.yaml"
PINS = {
    HISTORY: "sha256:1dc5476cdcb37f1a55e01259af4f4749eac8320776703b5f52466f3ce1f0d1c7",
    ARCHIVE: "sha256:a603c3b47b5fd7a22cd66a9191e11fc889f9a744a0750d65c8d9636a0359857d",
    QUEUE: "sha256:807f90eb662bdc9bc9a1e82e3b142a2bd599eb14b0b9e29565964f352e5b5d34",
}
reference, checksum, seal = prior.reference, prior.checksum, prior.seal
METHOD_ROWS = [
    dict(
        method="ImplicitHop",
        source="https://arxiv.org/abs/2610.41169",
        tasks=[8945, 8946, 8948, 8949, 8950],
        control="Equal shortest-distance answer-set cardinality; paired graph units.",
        limits="Prospective opportunity control; no proved explanation of old failure.",
    ),
    dict(
        method="Replay and gradient alignment",
        source="https://arxiv.org/abs/2508.01908",
        tasks=[8947, 8953, 8954, 8955],
        control="Released-label gradients, uniform/signature/repeated-current controls; charge scans.",
        limits="Local small-head adaptation; generator weights remain frozen.",
    ),
    dict(
        method="Reasoning or pattern exploitation",
        source="https://arxiv.org/abs/2610.39182",
        tasks=[8946, 8948, 8949, 8950],
        control="Label, edge-order and answer-position controls.",
        limits="Finite-game findings do not establish the same mechanism here.",
    ),
    dict(
        method="KG-Guard",
        source="https://arxiv.org/abs/2606.00328",
        tasks=[8951, 8952],
        control="Graph context versus confidence; no answer leakage.",
        limits="Future comparator only; deterministic truth is not learned detection.",
    ),
    dict(
        method="EBT",
        source="https://arxiv.org/abs/2507.02092",
        tasks=[8952],
        control="Keep generator, learned confidence and deterministic truth separate.",
        limits="No foundation-model reproduction or architecture change.",
    ),
    dict(
        method="ARM-EBM",
        source="https://arxiv.org/abs/2512.15605",
        tasks=[8952],
        control="Preserve generator and verifier roles.",
        limits="No transferred paper scores.",
    ),
    dict(
        method="Distributional EBMs",
        source="https://arxiv.org/abs/2605.18871",
        tasks=[8946, 8952],
        control="Frozen source-bound calibration and length controls.",
        limits="No oracle-derived detector features or new ensemble.",
    ),
    dict(
        method="Symbolic correctness",
        source="https://arxiv.org/abs/2608.14569",
        tasks=[8945, 8951],
        control="Independently recompute exact graph labels.",
        limits="Neural confidence is no correctness certificate.",
    ),
    dict(
        method="Learning-to-sample",
        source="https://arxiv.org/abs/2605.24752",
        tasks=[8947],
        control="Keep the normalized decision head; no added MCMC.",
        limits="Parameter learning proves no sampling efficiency.",
    ),
    dict(
        method="HalluScoring",
        source="https://arxiv.org/abs/2609.38355",
        tasks=[8952],
        control="Per-graph errors, calibration and source shift.",
        limits="No broad transfer claim.",
    ),
    dict(
        method="KAC",
        source="https://arxiv.org/abs/2503.21076",
        tasks=[8947],
        control="Keep representation fixed when retention already passes.",
        limits="Deferred spline architecture; no new causal factor.",
    ),
    dict(
        method="KAN forgetting",
        source="https://ojs.aaai.org/index.php/AAAI/article/view/33986",
        tasks=[8947],
        control="Preserve representation while testing selection.",
        limits="Deferred representation option only.",
    ),
    dict(
        method="Energy-guided decoding",
        source="https://arxiv.org/abs/2507.07731",
        tasks=[8949, 8950],
        control="Do not modify the error-source generator.",
        limits="Visual-object results do not establish text-source benefits.",
    ),
    dict(
        method="FPGA-ASIC co-design",
        source="https://arxiv.org/abs/2602.15985",
        tasks=[8955],
        control="Charge complete requests, persistence and candidate scans.",
        limits="Hardware availability and removable fraction remain unknown.",
    ),
    dict(
        method="Trust-region continual learning",
        source="https://arxiv.org/abs/2602.02417",
        tasks=[8947, 8953],
        control="Test selection information before another optimizer.",
        limits="No Fisher optimizer or claimed reproduction.",
    ),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so supervision can keep the original wall-clock budget."""
    print(f"[exp8944] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: str) -> Json:
    """Keep absent bytes separate from rejected producers and measured zeros."""
    return dict(
        check=check,
        upstream=TASK,
        artifact_field=field,
        expected=expected,
        observed=observed,
        **reference(path),
    )


def authority(root: Path) -> Json:
    """Bind complete independent objects; accept activation after staging is consumed."""
    tasks, table, rows, failures = [], [], [], []
    mode = "unavailable"
    path = root / DESIGN
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
        checked = compare_contract(text, *queues, milestone=MILESTONE, first_id=8944, count=13)
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
            and "<!-- V765_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("independent_authority_mismatch")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, yaml.YAMLError):
        failures.append(
            failure(
                path,
                "complete_authority",
                "full_task_objects",
                "13 exact ordered full objects, table, digest and queue",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        )
    for i in range(13):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=f"exp{8944 + i}-missing",
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
            absolute_metric=int(matched),
            denominator=1,
            cluster_id="administrative_contract",
            metric="ordered_task_contract_match",
            condition="exact_authority",
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
    """Each control changes one valid private queue, preserving causal rejection."""
    text = (source / DESIGN).read_text()
    queue = yaml.load(
        (
            source / (STAGED if authority(source)["comparison_mode"] == "staged" else ACTIVE)
        ).read_bytes(),
        Loader=yaml.CSafeLoader,
    )
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


def history(raw: Path) -> Json:
    """Preserve pinned producer bytes and actual skips without re-executing their science."""
    refs, failures, outcomes = [], [], []
    health, source_gate = {}, {}
    for name, pin in PINS.items():
        path = ROOT / name
        if reference(path)["sha256"] != pin:
            failures.append(
                failure(
                    path,
                    "original_v764",
                    "sha256",
                    pin,
                    "missing_file" if not path.is_file() else "rejected_producer",
                )
            )
            continue
        refs.append(seal(path, raw))
        if name == HISTORY:
            try:
                producer = json.loads(path.read_bytes())
                outcomes = producer["outcome_rows"]
                health = producer["repository_health"]
                if len(outcomes) != 14 or producer["experiment_id"] != 8943:
                    raise ValueError("historical_census")
            except (ValueError, KeyError, TypeError) as error:
                failures.append(
                    failure(
                        path,
                        "original_v764_schema",
                        "outcome_rows",
                        "original producer object",
                        "missing_field" if isinstance(error, KeyError) else "rejected_producer",
                    )
                )
    for index, row in enumerate(outcomes):
        ref = row.get("primary_reference")
        if ref:
            path = Path(ref["path"])
            if reference(path)["sha256"] != ref["sha256"]:
                failures.append(
                    failure(
                        path,
                        "original_retained_primary",
                        "sha256",
                        ref["sha256"],
                        "missing_file" if not path.is_file() else "rejected_producer",
                    )
                )
            else:
                refs.append(seal(path, raw))
                if row.get("experiment_id") == 8934:
                    producer = json.loads(path.read_bytes())
                    if "depth_source_ready_score" not in producer:
                        failures.append(
                            dict(
                                failure(
                                    path,
                                    "missing_field",
                                    "depth_source_ready_score",
                                    0,
                                    "missing_field",
                                ),
                                upstream=row["task_id"],
                            )
                        )
                        continue
                    source_gate = dict(
                        upstream=row["task_id"],
                        **ref,
                        check="observed_zero",
                        artifact_field="depth_source_ready_score",
                        expected=1,
                        observed=producer["depth_source_ready_score"],
                    )
        progress("historical_outcomes", index + 1, len(outcomes) - index - 1)
    if not failures:
        try:
            text = (ROOT / ARCHIVE).read_text()
            tasks = parse_design(text, milestone=prior.MILESTONE)[1]
            queue = yaml.load((ROOT / QUEUE).read_bytes(), Loader=yaml.CSafeLoader)
            for actual, expected in zip(queue["tasks"], tasks):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
            original = json.loads(Path(outcomes[0]["primary_reference"]["path"]).read_bytes())
            if not (
                canonical_hash(tasks) == prior.PIN
                and tasks == queue["tasks"] == original["full_task_objects"]
                and compare_contract(
                    text, queue, queue, milestone=prior.MILESTONE, first_id=8930, count=14
                )["passed"]
                and [r["task_id"] for r in outcomes] == [t["id"] for t in tasks]
                and source_gate["observed"] == 0
            ):
                raise ValueError("retained_authority_mismatch")
        except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
            failures.append(
                failure(
                    ROOT / ARCHIVE,
                    "original_authority",
                    "full_task_objects",
                    "original V764 objects and failed gate",
                    "missing_field" if isinstance(error, KeyError) else "rejected_producer",
                )
            )
    for row in failures:
        if row["upstream"] == TASK:
            row["upstream"] = "exp8943-capstone"
    return dict(
        authenticated=not failures,
        references=refs,
        failures=failures,
        outcome_rows=outcomes,
        failed_source_gate=source_gate,
        repository_health=health,
        scientific_results_admitted=False,
        compute_scope="Opaque original producer references; no current compute credit.",
    )


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Freeze each invocation before validation without model or device setup."""
    progress("preconditions_before", 0, 13)
    if (raw / "work.json").exists():
        raise FileExistsError("immutable_invocation_already_exists")
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    historical = history(raw)
    snapshots, failures = list(historical["references"]), []
    for name in [DESIGN, ACTIVE, STAGED, LITERATURE]:
        path = root / name
        if path.is_file():
            snapshots.append(dict(seal(path, raw), current_execution_authority=name != LITERATURE))
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        elif name != STAGED:
            failures.append(
                failure(path, "authority_input", name, "original bytes", "missing_file")
            )
    for name in [
        *OWNED,
        TEST,
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/v755_branch_contract.py",
        "python/carnot/reporting/v755_branch_runner.py",
        "python/carnot/reporting/v764_branch_contract.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v759_branch_contract.py",
        "python/carnot/reporting/v756_branch_qualification.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/experiment_template.py",
        "ops/exclusion_manifest.yaml",
    ]:
        snapshots.append(dict(seal(ROOT / name, raw), executed_source=True))
    contract = authority(raw / "authority")
    controls = mutations(raw / "authority", raw / "controls") if contract["ready"] else []
    work = dict(
        contract=contract,
        history=historical,
        methods=deepcopy(METHOD_ROWS),
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
            no_model_loads=True,
            no_device_probes=True,
            activation_authorized=False,
            generator_weights_frozen=True,
            arc_suspended=True,
            run_date_matches=run_date == "20261109",
            repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
            free_bytes=shutil.disk_usage(raw).free,
            required_free_bytes=536870912,
            time_budget_s=4200,
            hard_cap_s=4800,
        ),
        raw_operands=[reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()],
    )
    atomic_json(raw / "work.json", work)
    progress(
        "preconditions_after",
        sum(r["matched"] for r in contract["rows"]),
        13 - sum(r["matched"] for r in contract["rows"]),
    )
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Reuse the current receipt reducer; only exact custody earns readiness."""
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
        history=dict(work["history"], original_child_qualified=work["history"]["authenticated"]),
        branches=dict(runtime=dict(ready=False), arc=dict(ready=False), failures=[]),
    )
    value = base.build(prepared, validation)
    tasks, resources, controls = (
        work["contract"]["tasks"],
        work["preconditions"],
        work["authority_mutation_rows"],
    )
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
        g["upstream"] != TASK and g["artifact_field"] != "contract_ready_score"
        for t in tasks
        for g in t["gated_on"]
    ) and all(not t.get("requires") for t in tasks)
    checked = bool(
        value["required_checks_passed"]
        and resources["repository_virtualenv"]
        and resources["run_date_matches"]
        and resources["free_bytes"] >= resources["required_free_bytes"]
        and value["duration_s"] < 4200
        and all(r["passed"] for r in controls)
    )
    ready = bool(
        checked
        and work["contract"]["ready"]
        and work["history"]["authenticated"]
        and not work["input_failures"]
        and len(controls) == 5
        and independent
        and all(r["required_field_declared"] for r in gates)
    )
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    for field in [
        "preserved_v754_reference",
        "runtime_repair_ready_score",
        "arc_new_outcomes_available_score",
        "branch_admission_rows",
    ]:
        value.pop(field)
    value.update(
        experiment_id=8944,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7658944,
        honest_verdict=f"complete_{verdict}_v765_branch_contract",
        verdict_class=verdict,
        contract_ready_score=int(ready),
        required_checks_passed=checked,
        model_specs=[],
        rows=deepcopy(work["contract"]["rows"]),
        contract_rows=work["contract"]["rows"],
        intended_count=13,
        censored_count=13 - value["completed_count"],
        full_task_digest=canonical_hash(tasks),
        canonical_tasks_sha256=PIN,
        execution_snapshot=[s for s in work["snapshots"] if s.get("current_execution_authority")],
        preserved_v764_reference=work["history"],
        authority_comparison_mode=work["contract"]["comparison_mode"],
        source_method_rows=work["methods"],
        prior_scope_rows=[dict(task_id=t["id"], **p) for t in tasks for p in t["prior_failures"]],
        authority_mutation_rows=controls,
        gate_edge_rows=gates,
        execution_venue="host",
        preconditions_checked=resources,
        gate_check_summary=work["contract"]["failures"]
        + work["history"]["failures"]
        + work["input_failures"],
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            owned_validation=checked,
            preserved_history=work["history"]["authenticated"],
            independent_cpu_roots=independent,
            no_retired_runtime_requires=all(not t.get("requires") for t in tasks),
            exact_gate_fields=all(r["required_field_declared"] for r in gates),
            mutation_controls=len(controls) == 5 and all(r["passed"] for r in controls),
        ),
        code_config_hashes=[s for s in work["snapshots"] if s.get("executed_source")],
        methodology_note="Administrative custody only. The old depth-source gate remains failed. Matched opportunity and gradient selection are proposed controls, not measured benefits. CPU science admits its own inputs. Historical outcomes and actual scheduler skips retain their original verdicts. No activation, device probes, model work or external publication.",
    )
    if verdict == "blocked":
        value["inference_substrate_class"] = "blocked_no_run"
        value["current_work_receipt"]["inference_substrate_class"] = "blocked_no_run"
    value["raw_shard_hashes"] += work["raw_operands"]
    value["reproduction_commands"] = [
        f"PYTHONPATH=python .venv/bin/python -u {CLI} --date {work['run_date']}",
        f".venv/bin/python -u {CLI} --cold-replay <primary-path>",
    ]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Exact immutable custody and owned checks establish administrative readiness only. Contract, prior and literature rows are receipts outside comparative rows. Current compute is zero; upstream evidence remains opaque and proves no new detector value or independent natural-language generalization.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild reported claims from sealed operands, including rehashed authority edits."""
    try:
        value = json.loads(path.read_bytes())
        if value["required_checks_passed"] and not verify_measured(
            value["owned_statement_coverage"], OWNED
        ):
            raise ValueError("unqualified_statements")
        for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        ref = value["raw_shard_hashes"][0]
        work = json.loads(Path(ref["path"]).read_bytes())
        if (
            work["contract"] != authority(Path(work["raw"]) / "authority")
            or work["methods"] != METHOD_ROWS
        ):
            raise ValueError("authority_drift")
        for saved in work["history"]["references"]:
            pin = next(
                (p for n, p in PINS.items() if str(ROOT / n) == saved["source"]["path"]), None
            )
            if pin and (saved["sha256"] != pin or reference(Path(saved["path"]))["sha256"] != pin):
                raise ValueError("historical_pin_drift")
            if saved["source"]["path"] == str(ROOT / HISTORY):
                if (
                    work["history"]["outcome_rows"]
                    != json.loads(Path(saved["path"]).read_bytes())["outcome_rows"]
                ):
                    raise ValueError("history_drift")
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
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
