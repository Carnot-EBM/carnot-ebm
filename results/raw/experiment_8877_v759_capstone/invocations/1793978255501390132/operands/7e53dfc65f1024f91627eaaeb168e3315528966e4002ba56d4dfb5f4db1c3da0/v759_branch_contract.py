"""REQ-REPORT-8866: missing independent authority cannot become a positive receipt."""

from copy import deepcopy
from datetime import UTC, datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v755_branch_contract as base
from carnot.reporting import v758_branch_contract as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import compare_contract, parse_design
from carnot.reporting.v685_authority_lifecycle import _snapshot
from carnot.reporting.v749_contract_methods import encode
from carnot.reporting.v756_branch_qualification import verify_measured
from scripts.audit_roadmap_gates import _required_artifact_fields_block
from scripts.experiment_template import normalize_artifact_for_template_write
from scripts.roadmap_schema import Roadmap

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = prior.ROOT, prior.DESIGN, prior.ACTIVE, prior.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8866_v759_branch_contract",
    "exp8866-branch-contract",
    "2026.11.759",
)
PIN = "f9876c16be76ba17de332ec20d9998342c48841912bc9db6a2cce910d8730bf2"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/v759_branch_contract.py", CLI]
TEST = "tests/python/test_v759_branch_contract_8866.py"
ARCHIVE = "openspec/change-proposals/research-roadmap-v758-preserved-20261106.md"
PRIOR = "results/experiment_8853_v758_branch_contract.json"
CAPSTONE = "results/experiment_8865_v758_capstone.json"
PINS = {
    PRIOR: "772046dd8cd218673e9882e232faf9fd5bda0de7ad257f7493564e7591b99c87",
    CAPSTONE: "e7ea48c9366597069f664263543f07660501de5575ea723518422e6bdcd380d6",
    ARCHIVE: "b947c7b2712e9b75aef6cf127a03d1243e9ce761d548ea5b39192b4cca94273a",
}
reference, checksum = prior.reference, prior.checksum


def seal(source: Path, raw: Path) -> Json:
    """Reuse immutable snapshots without growing historical filenames at each rollover."""
    saved = _snapshot(source, source.read_bytes(), raw / "inputs", "operand")
    return dict(source=reference(source), path=saved["snapshot_path"], sha256=saved["sha256"])


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so supervising processes can enforce the fixed deadline."""
    print(f"[exp8866] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, check: str, field: str, expected: Any, observed: Any) -> Json:
    """Absent evidence differs from rejected bytes and a measured zero."""
    return dict(
        check=check,
        upstream=TASK,
        artifact_field=field,
        expected=expected,
        observed=observed,
        **reference(path),
    )


def authority(root: Path) -> Json:
    """Keep queue observations when independent JSON is absent, without asserting equality."""
    tasks, table, rows, failures = [], [], [], []
    try:
        active = yaml.safe_load((root / ACTIVE).read_bytes())
        staged = (
            yaml.safe_load((root / STAGED).read_bytes())
            if (root / STAGED).is_file()
            else deepcopy(active)
        )
        if active["milestone"] == "2026.11.758" and (root / STAGED).is_file():
            active = deepcopy(staged)
        for queue in [staged, active]:
            Roadmap.model_validate(queue)
        tasks = deepcopy(staged["tasks"])
        text = (root / DESIGN).read_text()
        table, machine = parse_design(text, milestone=MILESTONE)
        for queue in [staged, active]:
            for actual, expected in zip(queue["tasks"], machine):
                if actual == dict(expected, prompt=encode(expected["prompt"])):
                    actual["prompt"] = expected["prompt"]
        checked = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8866, count=12
        )
        rows = checked["rows"]
        for i, row in enumerate(rows):
            row["checks"]["full_object"] = (
                i < min(len(machine), len(staged["tasks"]), len(active["tasks"]))
                and machine[i] == staged["tasks"][i] == active["tasks"][i]
            )
            row["matched"] = all(row["checks"].values())
        tasks = machine
        if not (
            checked["passed"]
            and all(r["matched"] for r in rows)
            and canonical_hash(tasks) == "sha256:" + PIN
            and f"Canonical task SHA-256: `{PIN}`" in text
            and "<!-- V759_TASK_CONTRACT_END -->" in text
        ):
            raise ValueError("rejected_producer")
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        path = root / DESIGN
        observed = (
            "missing_file"
            if not path.is_file()
            else "missing_field"
            if isinstance(error, IndexError)
            else "rejected_producer"
        )
        failures.append(
            failure(
                path,
                "complete_authority",
                "full_task_objects",
                "12 independent full objects, visible table, digest and closing marker",
                observed,
            )
        )
    for i in range(12):
        if i >= len(rows):
            rows.append(
                dict(
                    unit_id=tasks[i]["id"] if i < len(tasks) else f"exp{8866 + i}-missing",
                    order=i + 1,
                    checks=dict(independent_authority=False),
                    matched=False,
                )
            )
        row = rows[i]
        matched = row["matched"] and not failures
        row.update(
            matched=matched,
            metric=int(matched),
            absolute_metric=int(matched),
            denominator=1,
            raw_numerator=sum(row["checks"].values()),
            raw_denominator=len(row["checks"]),
            cluster_id="administrative_contract",
            arm="full_object_custody",
            seed=None,
            status="completed" if matched else "censored",
            censored=not matched,
            missing_reason=None if matched else "independent_authority_unavailable_or_rejected",
            excluded=False,
            effective_independent_groups=0,
        )
    return dict(ready=not failures, rows=rows, tasks=tasks, table=table, failures=failures)


def mutations(source: Path, raw: Path) -> list[Json]:
    """Repeat the seven original V758 corruptions only on private copies."""
    text = (source / DESIGN).read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    queue = yaml.safe_load((source / ACTIVE).read_bytes())
    rows = []
    for mode in [
        "missing_last_task",
        "truncation",
        "prompt",
        "gate",
        "model",
        "closing_marker",
        "table",
    ]:
        root = raw / mode
        design = root / DESIGN
        design.parent.mkdir(parents=True, exist_ok=True)
        changed, body = deepcopy(tasks), text
        if mode == "missing_last_task":
            changed.pop()
        if mode == "truncation":
            body = text[: text.index('"id": "exp8877-')]
        if mode in {"prompt", "model"}:
            changed[0][mode] = "forged"
        if mode == "gate":
            changed[3]["gated_on"][0]["artifact_field"] = "wrong_gate"
        if mode == "closing_marker":
            body = text.replace("<!-- V759_TASK_CONTRACT_END -->", "")
        if mode == "table":
            body = text.replace("| 1 | exp8866-branch-contract", "| 2 | exp8866-branch-contract")
        design.write_text(body)
        (root / ACTIVE).write_text(yaml.safe_dump(dict(queue, milestone=MILESTONE, tasks=changed)))
        rows.append(dict(mutation=mode, passed=not authority(root)["ready"]))
        progress("authority_control_" + mode, len(rows), 7 - len(rows))
    return rows


def history(raw: Path) -> Json:
    """Seal consumed original operands; capstone summaries retain their original verdicts."""
    refs, failures, verdicts = [], [], {}
    for name, pin in PINS.items():
        path = ROOT / name
        if reference(path)["sha256"] != "sha256:" + pin:
            failures.append(
                failure(
                    path,
                    "preserved_v758",
                    "sha256",
                    pin,
                    "missing_file" if not path.is_file() else "rejected_producer",
                )
            )
            continue
        refs.append(seal(path, raw))
        if name == ARCHIVE:
            continue
        value = json.loads(path.read_bytes())
        verdicts[name] = value["verdict_class"]
        operands = value["code_config_hashes"] + (
            value["raw_shard_hashes"]
            if name == PRIOR
            else [
                r["primary_reference"]
                for r in value["rows"]
                if r.get("primary_reference", {}).get("sha256")
            ]
            + [s["reference"] for r in value["rows"] for s in r.get("scheduler_receipts", [])]
        )
        for i, ref in enumerate({r["path"]: r for r in operands}.values()):
            operand = Path(ref["path"])
            if reference(operand)["sha256"] != ref["sha256"]:
                failures.append(
                    failure(
                        operand,
                        "original_source_raw_operand",
                        "sha256",
                        ref["sha256"],
                        "missing_file" if not operand.is_file() else "rejected_producer",
                    )
                )
            else:
                refs.append(seal(operand, raw))
            if i % 20 == 0:
                progress("historical_operands", i + 1, len(operands) - i - 1)
    return dict(
        authenticated=not failures,
        references=refs,
        failures=failures,
        verdict_class=verdicts.get(PRIOR),
        prior_verdicts=verdicts,
        scientific_results_admitted=False,
        historical_replay_required=False,
    )


def runtime(root: Path, raw: Path) -> Json:
    """Reuse Exp8853's causal reader and exact custody checks with the current date."""
    path = root / "ops/cuda-runtime-repair-receipt.json"
    checks: Json = {}
    refs = []
    if path.is_file():
        refs.append(seal(path, raw))
        try:
            value = json.loads(path.read_bytes())
            admitted = prior.read_runtime(root)
            checks["original_causal_validator"] = admitted["ready"]
            checks["new_dated_library_environment_change"] = (
                value["run_date"] == "20261106"
                and datetime.fromtimestamp(value["executed_at_ns"] / 1e9, UTC).strftime("%Y%m%d")
                == value["run_date"]
                and bool(value["description"])
                and any(
                    r["field"] in {"libraries", "environment"} and r["changed"]
                    for r in admitted.get("changes", [])
                )
            )
            model = value["mandated_model_provenance"]
            checks["exact_model"] = (
                model["hf_id"] == "unsloth/Qwen3.8-27B-GGUF"
                and model["revision"] == prior.MODEL_REVISION
                and model["revision"] in model["path"]
                and model["sha256"] == prior.MODEL_PIN
                and reference(Path(model["path"]))["sha256"] == model["sha256"]
            )
            validators = value["validator_references"]
            checks["current_validator_custody"] = {r["source_path"] for r in validators} == set(
                prior.RUNTIME_VALIDATORS
            ) and all(
                reference(ROOT / r["source_path"])["sha256"] == r["sha256"] for r in validators
            )
            for key in ["current_reference", "previous_reference", "transcript_reference"]:
                ref = value[key]
                if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                    raise ValueError("runtime_operand_drift")
                refs.append(seal(Path(ref["path"]), raw))
            transcript = json.loads(Path(value["transcript_reference"]["path"]).read_bytes())
            checks["actual_context_copy_offload"] = (
                admitted["ready"]
                and transcript["model_provenance"] == model
                and transcript["offloaded_layers"] > 0
            )
        except (OSError, ValueError, KeyError, TypeError):
            checks["complete_supplied_evidence"] = False
    ready = bool(checks) and all(checks.values())
    failures = (
        []
        if ready
        else [
            failure(
                path,
                "authenticated_changed_runtime",
                "runtime_repair_ready_score",
                "new causal changes, current validators, exact GGUF and context/copy/offload",
                "missing_file" if not path.is_file() else "rejected_producer",
            )
        ]
    )
    return dict(
        ready=ready,
        checks=checks,
        references=refs,
        gate_check_summary=failures,
        model_loads=0,
        device_probes=0,
    )


def methods(root: Path) -> list[Json]:
    """Ingest the supplied primary map without importing external scientific guarantees."""
    text = (root / DESIGN).read_text()
    section = text.split("## Research basis", 1)[1].split("## V759 architecture", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [v.strip() for v in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0].startswith("["):
            rows.append(
                dict(
                    source=cells[0],
                    adopted_method=cells[1],
                    deferred_claim=cells[2],
                    paper_results_inherited=False,
                )
            )
    return rows


def freeze(root: Path, raw: Path, run_date: str) -> Json:
    """Check private resources and retain original bytes before validation starts."""
    progress("preconditions_before", 0, 12)
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    probe = raw / "write_probe"
    probe.write_text(TASK)
    writable = probe.read_text() == TASK
    probe.unlink()
    resources = dict(
        repository_virtualenv=Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(),
        private_output_writable=writable,
        private_mode=oct(raw.stat().st_mode & 0o777),
        free_bytes=shutil.disk_usage(raw).free,
        required_free_bytes=536870912,
        installed_dependencies={
            p: importlib.metadata.version(p) for p in ["coverage", "pytest", "pydantic", "PyYAML"]
        },
        time_budget_s=4200,
        no_model_loads=True,
        no_device_probes=True,
        activation_authorized=False,
    )
    contract = authority(root)
    historical = history(raw)
    branches = runtime(root, raw)
    snapshots = historical["references"] + branches["references"]
    for name in [DESIGN, ACTIVE, STAGED]:
        if (root / name).is_file():
            snapshots.append(seal(root / name, raw))
            target = raw / "authority" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / name).read_bytes())
    for name in ["research-references.md", "ops/exclusion_manifest.yaml"]:
        snapshots.append(seal(ROOT / name, raw))
    contract = authority(raw / "authority")
    controls = mutations(root, raw / "controls") if contract["ready"] else []
    work = dict(
        contract=contract,
        history=historical,
        branches=branches,
        methods=methods(root) if (root / DESIGN).is_file() else [],
        snapshots=snapshots,
        authority_mutation_rows=controls,
        run_date=run_date,
        owner_pid=os.getpid(),
        start_ns=start,
        end_ns=time.monotonic_ns(),
        raw=str(raw.absolute()),
        preconditions=resources,
        raw_operands=[reference(p) for p in sorted((raw / "controls").rglob("*")) if p.is_file()],
    )
    atomic_json(raw / "work.json", work)
    progress(
        "preconditions_after",
        sum(r["matched"] for r in contract["rows"]),
        12 - sum(r["matched"] for r in contract["rows"]),
    )
    return dict(work, raw_reference=reference(raw / "work.json"))


def build(work: Json, validation: Json) -> Json:
    """Owned failures disqualify; missing external authority stays terminal blocked."""
    prepared = dict(
        work,
        history=dict(work["history"], original_child_qualified=work["history"]["authenticated"]),
        manifests=[],
        branches=dict(runtime=work["branches"], arc=dict(ready=False), failures=[]),
    )
    prepared["end_ns"] = max(
        [
            work["end_ns"],
            *[r.get("ended_monotonic_ns", work["end_ns"]) for r in validation["receipts"]],
        ]
    )
    value = base.build(prepared, validation)
    tasks = work["contract"]["tasks"]
    producers = {t["id"]: t for t in tasks}
    gates = [
        dict(
            downstream=t["id"],
            **g,
            required_field_declared=g["upstream"] in producers
            and g["artifact_field"]
            in _required_artifact_fields_block(producers[g["upstream"]]["prompt"]),
        )
        for t in tasks
        for g in t.get("gated_on", [])
    ]
    independent = all(
        g["upstream"] != TASK
        or t["id"] == "exp8875-qwen-source-canary"
        and g["artifact_field"] == "runtime_repair_ready_score"
        for t in tasks
        for g in t.get("gated_on", [])
    )
    resources, controls = work["preconditions"], work["authority_mutation_rows"]
    checked = bool(
        value["required_checks_passed"]
        and resources["private_output_writable"]
        and resources["repository_virtualenv"]
        and resources["free_bytes"] >= resources["required_free_bytes"]
        and all(r["passed"] for r in controls)
    )
    ready = bool(
        checked
        and work["contract"]["ready"]
        and work["history"]["authenticated"]
        and len(controls) == 7
        and len(work["methods"]) == 6
        and all(r["required_field_declared"] for r in gates)
        and independent
    )
    verdict = "disqualified" if not checked else "null" if ready else "blocked"
    failures = (
        work["contract"]["failures"]
        + work["history"]["failures"]
        + work["branches"]["gate_check_summary"]
    )
    value.pop("preserved_v754_reference")
    value.pop("arc_new_outcomes_available_score")
    value.update(
        experiment_id=8866,
        task_id=TASK,
        milestone=MILESTONE,
        random_seed=7598866,
        honest_verdict=f"complete_{verdict}_v759_branch_contract",
        verdict_class=verdict,
        intended_count=12,
        censored_count=12 - value["completed_count"],
        contract_ready_score=int(ready),
        runtime_repair_ready_score=int(work["branches"]["ready"]),
        canonical_tasks_sha256=PIN,
        required_checks_passed=checked,
        execution_venue="host",
        preconditions_checked=resources,
        gate_check_summary=failures,
        gate_producer_rows=gates,
        authority_mutation_rows=controls,
        source_method_map=work["methods"],
        preserved_v758_reference=work["history"],
        external_admission_rows=[
            dict(
                branch="runtime",
                downstream="exp8875-qwen-source-canary",
                optional=True,
                ready=work["branches"]["ready"],
                independent_cpu_roots_gated=False,
            )
        ],
        acceptance_gates=dict(
            exact_contract=work["contract"]["ready"],
            owned_validation=checked,
            preserved_history=work["history"]["authenticated"],
            mutation_controls=len(controls) == 7 and all(r["passed"] for r in controls),
            method_map=len(work["methods"]) == 6,
            exact_gate_fields=all(r["required_field_declared"] for r in gates),
            independent_cpu_roots=independent,
        ),
        code_config_hashes=[
            reference(ROOT / p)
            for p in OWNED
            + [
                TEST,
                "python/carnot/reporting/v758_branch_contract.py",
                "python/carnot/reporting/v755_branch_contract.py",
                "python/carnot/reporting/v755_branch_runner.py",
                *prior.REUSED,
            ]
        ],
        arc_task_count=sum("arc" in t["title"].lower().split() for t in tasks),
        generator_training_task_count=sum(
            "generator training" in t["title"].lower() for t in tasks
        ),
        methodology_note="Administrative equality and supplied method ingestion only. Queue observations cannot replace absent design JSON. No activation, science, model load, ARC or generator training.",
    )
    value["phase_spans"] = [
        dict(phase="operand_custody", start_s=0, end_s=(work["end_ns"] - work["start_ns"]) / 1e9),
        dict(
            phase="owned_validation",
            start_s=(work["end_ns"] - work["start_ns"]) / 1e9,
            end_s=value["duration_s"],
        ),
    ]
    value["raw_shard_hashes"] += work["raw_operands"]
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Administrative custody is not scientific benefit. Observed queue objects are unqualified until independent design equality; external absence blocks, owned failure disqualifies, optional runtime never gates CPU roots.",
    )
    value = normalize_artifact_for_template_write(value)
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Reconstruct authority and reduction from sealed operands, rejecting rehashed claims."""
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
        frozen = Path(work["raw"]) / "authority"
        if work["contract"] != authority(frozen):
            raise ValueError("authority_drift")
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
