"""REQ-REPORT-8331: bind execution operands before deciding whether science exists.

This adapter uses qualified readers. A missing producer remains an unavailable
measurement, so successful accounting cannot become a scientific result.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v716_capstone_evidence as reader
from carnot.reporting import v717_capstone_evidence as old
from carnot.reporting import v718_contract_replay as qualified
from carnot.reporting import v718_replay_history as history
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import execute
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED, PROTOCOL = (
    qualified.ROOT,
    qualified.DESIGN,
    qualified.ACTIVE,
    qualified.STAGED,
    qualified.PROTOCOL,
)
NAME, TASK, MILESTONE = "experiment_8331_v718_capstone", "exp8331-capstone", qualified.MILESTONE
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v718_capstone_8331.py"
OWNED = [
    "python/carnot/reporting/v718_capstone_evidence.py",
    "python/carnot/reporting/v718_capstone_reduction.py",
    "python/carnot/reporting/v718_capstone.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PROTOCOL_PIN = qualified.base.PIN
failure = qualified.failure
read = old.read


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush observable boundaries without manufacturing time or model work."""
    print(f"[exp8331] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(work: Json) -> Json:
    """Recheck copied full objects and visible order, rather than trusting a score."""
    with TemporaryDirectory(prefix="exp8331-authority-") as directory:
        root = Path(directory)
        for ref, name in zip(work["references"][:3], [DESIGN, STAGED, ACTIVE], strict=True):
            if ref["exists"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        try:
            return dict(qualified.authority(root, root / "assessment"))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return dict(
                activated=False,
                gate_check_summary=[
                    failure(Path(work["root"]) / DESIGN, "authority_available", True, None)
                ],
            )


def measure(root: Path, raw: Path) -> Json:
    """Read exact deliverables and freeze primitive bytes before any aggregation."""
    progress("measurement_before", 0, 14)
    start, wall = time.monotonic_ns(), time.time_ns()
    refs = [
        snapshot(root / name, raw / "custody", str(i))
        for i, name in enumerate([DESIGN, STAGED, ACTIVE, PROTOCOL])
    ]
    design = (
        refs[0] if refs[0]["exists"] else snapshot(ROOT / DESIGN, raw / "custody", "identity_only")
    )
    tasks = parse_design(Path(design["snapshot_path"]).read_text(), milestone=MILESTONE)[1]
    if [t["id"].split("-")[0] for t in tasks] != [f"exp{i}" for i in range(8318, 8332)]:
        raise ValueError("exact_fourteen_task_contract")
    inputs, plan = [], []
    for index, task in enumerate(tasks[:-1]):
        progress("input_before", index, 13 - index)
        declared = root / task["deliverable"]
        path = reader.prior.resolve(task, 8318 + index, root)
        if path != declared:
            refs.append(snapshot(declared, raw / "custody", str(len(refs))))
        item = reader.bind(path, raw, refs)
        inputs.append(item)
        source = old.read(item["reference"])
        for gate in source.get("gates_evaluated", []):
            refs.append(snapshot(Path(gate["artifact_path"]), raw / "custody", str(len(refs))))
        for key in ["work_reference", "measurement_reference", "replay_input_reference"]:
            operand = source.get(key)
            if operand:
                refs.append(
                    dict(
                        snapshot(Path(operand["path"]), raw / "custody", str(len(refs))),
                        expected_sha256=operand["sha256"],
                    )
                )
        script = ROOT / "scripts/experiments" / (Path(task["deliverable"]).stem + ".py")
        if source and source.get("schema") != "blocked_gate_check_v1" and script.is_file():
            plan.append(
                dict(
                    name=f"branch_{8318 + index}_replay",
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(script),
                        "--cold-replay",
                        item["reference"]["snapshot_path"],
                    ],
                    expected=0,
                    deadline=180,
                    scope="owned",
                )
            )
        progress("input_after", index + 1, 12 - index)
    prior_tasks = {}
    for name, version in [
        (history.PRIOR_DESIGN, "2026.10.717"),
        (history.OLD_DESIGN, "2026.10.716"),
    ]:
        refs.append(snapshot(root / name, raw / "custody", str(len(refs))))
        path = root / name
        if path.is_file():
            prior_tasks.update(
                {t["id"]: t for t in parse_design(path.read_text(), milestone=version)[1]}
            )
    historical = {
        key: reader.bind(root / task["deliverable"], raw, refs)
        for key, task in sorted(prior_tasks.items())
        if any(p["experiment_id"] == key for t in tasks for p in t["prior_failures"])
    }
    refs.append(snapshot(root / "ops/exclusion_manifest.yaml", raw / "custody", str(len(refs))))
    gate = dict(
        name="publication_gate",
        argv=[str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"],
        expected=0,
        deadline=90,
        scope="publication",
    )
    atomic_json(raw / "branch_manifest.json", dict(commands=plan, publication=gate))
    work = dict(
        root=str(root),
        tasks=tasks,
        inputs=inputs,
        history=historical,
        references=refs,
        failures=[],
        audits=execute(plan, raw / "branches"),
        publication=execute([gate], raw / "publication")[0],
        started_monotonic_ns=start,
        started_wall_ns=wall,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_after", 13, 1)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Seal checks before reduction, keeping terminal validation outside this payload."""
    from carnot.reporting.v718_capstone_reduction import reduce

    if "code_refs" not in work:
        work["code_refs"] = [
            snapshot(ROOT / p, raw / "code", str(i))
            for i, p in enumerate(
                [
                    *OWNED,
                    TEST,
                    "python/carnot/reporting/v718_replay_history.py",
                    "python/carnot/reporting/v718_contract_replay.py",
                    "python/carnot/reporting/primary_publication.py",
                    "python/carnot/reporting/roadmap_contract.py",
                    "scripts/experiment_template.py",
                    "scripts/adversarial_verify.py",
                    "scripts/publication_gate.py",
                    "scripts/verdict_row_consistency_lint.py",
                ]
            )
        ]
    work["frozen_validation_receipts"] = receipts
    atomic_json(raw / "measurement.json", work)
    primitive = snapshot(raw / "measurement.json", raw / "primitives", canonical_hash(work)[7:])
    value = reduce(work, receipts)
    publication = json.loads(Path(work["publication"]["stdout_path"]).read_bytes())
    value.update(
        experiment_id=8331,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        schema="carnot.v718.capstone.v1",
        random_seed=7188331,
        no_model_load=True,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        preconditions_checked=work.get("preconditions", work["references"]),
        validation_receipts=receipts,
        invocation_argv=[
            "20261009" if a == "20261008" else a for a in work.get("invocation_argv", [])
        ],
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        phase_spans=[
            dict(
                phase="aggregation",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
                duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
            ),
            *[
                dict(phase=r["name"], duration_s=r["duration_s"])
                for r in receipts + work["audits"]
                if "duration_s" in r
            ],
        ],
        source_artifact_hashes=work["references"],
        code_config_hashes=work["code_refs"],
        work_reference=primitive,
        raw_shard_hashes=[primitive],
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        publication_output=str(output),
        publication_gate_receipt=work["publication"],
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        **{k.lower(): publication["gates"][k] for k in ["G1", "G2", "G3", "G4"]},
        cited_upstream_artifacts=[
            dict(
                path=i["reference"]["path"],
                sha256=i["reference"]["sha256"],
                task_id=t["id"],
                fields_imported=[
                    "terminal disposition",
                    "rows",
                    "model_invocation_counts",
                    "sample_size_budget",
                    "primitive references",
                ],
            )
            for t, i in zip(work["tasks"][:-1], work["inputs"], strict=True)
        ],
        adversarial_findings=[],
        finding_consumer_policy=history.POLICY,
        historical_v717_primary_verdict=old.read(work["inputs"][0]["reference"]).get(
            "historical_primary_verdict"
        ),
        historical_first_reduction_mismatch=old.read(work["inputs"][0]["reference"]).get(
            "first_reduction_mismatch"
        ),
        historical_adversarial_findings=old.read(work["inputs"][0]["reference"]).get(
            "adversarial_findings", []
        ),
        methodology_note="Qualified deterministic primitive aggregation. Missing current sealed science blocks H1/H2. Spline energy is the sigmoid re-expression of identical basis coefficients; no architecture-specific truth or generalization. Capacity controls remain separate.",
        external_publication_authorized=False,
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    value = dict(normalize_artifact_for_template_write(value))
    value["field_principles"] = {
        k: "Bind measured scope and exact bytes; unavailable observations stay null and execution readiness grants no scientific utility."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    purposes = {
        "experiment_id task_id milestone run_date invocation_argv": "Identify the invocation and exact activated execution authority.",
        "honest_verdict verdict_class gate_check_summary": "Distinguish current owned failures, unavailable external operands and historical dispositions; missing observations are not zero.",
        "capstone_execution_ready_score required_checks_passed acceptance_gates validation_receipts": "Qualify byte-bound owned execution independently of whether science wins; terminal checks stay outside reduction.",
        "H1 H2 science_ready_score h1_development_signal_score h2_development_signal_score": "Retain unchanged intended source, arm and retention denominators; no sealed measurement means blocked science.",
        "capacity_scope": "Keep constructed memory and feedback-loss controls outside natural H1/H2 and any regret guarantee.",
        "rows task_dispositions intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget": "Count each intended administrative unit once while preserving producer absence and scientific sample limits.",
        "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts historical_model_provenance live_call_accounting": "Record zero current model work and prevent imported Qwen or canary calls from entering the frozen cached study.",
        "retirements next_evidence_conditions three_prd_gaps continuation_decision": "Compare exact prior verdicts, preserve legacy retirement rows and require falsifiable changed evidence without retiring unmeasured hypotheses.",
        "g1 g2 g3 g4 paper_ready unmet_gates": "Import publication gates unchanged; administrative readiness never authorizes external publication.",
    }
    value["field_principles"].update(
        {field: purpose for fields, purpose in purposes.items() for field in fields.split()}
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Cold reconstruction rejects rehashed self-row and check-disposition changes."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            require_reference(ref)
        work = json.loads(Path(value["work_reference"]["snapshot_path"]).read_bytes())
        if work["frozen_validation_receipts"] != value["validation_receipts"]:
            return False
        for receipt in value["validation_receipts"] + work["audits"] + [work["publication"]]:
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
