"""REQ-REPORT-8360: exact deployment authority never grants scientific benefit."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v718_contract_replay as legacy
from carnot.reporting import v718_replay_history as history
from carnot.reporting import v720_frozen_inputs as inputs
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED = legacy.ROOT, legacy.DESIGN, legacy.ACTIVE, legacy.STAGED
NAME, TASK, MILESTONE = (
    "experiment_8360_v721_contract_methods",
    "exp8360-contract-methods",
    "2026.10.721",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v721_contract_methods_8360.py"
METHODS = "openspec/change-proposals/v721-methods-manifest.json"
PROTOCOL = "openspec/change-proposals/v721-deployment-protocol.json"
METHODS_PIN = "sha256:4c7cc722c3e7859e4e4ec1d66235638d1c1552397b71da0c7f928e621690f684"
DEPLOYMENT_PIN = "sha256:d4441f7619a1d349a4958038af93c36f2c4df97b31fd32278c4ea6def7c7b4eb"
OWNED = [
    "python/carnot/reporting/v721_contract_methods.py",
    "python/carnot/reporting/v721_contract_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
failure = legacy.failure


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts distinguish bounded evidence reading from a stalled process."""
    print(f"[exp8360] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path, raw: Path, milestone: str = MILESTONE) -> Json:
    """The existing full-task reader accepts only already activated authority."""
    tasks = parse_design((root / DESIGN).read_text(), milestone=milestone)[1]
    staged = root / STAGED
    result = dict(
        assess_authorities(
            root / DESIGN,
            staged if staged.exists() else root / ACTIVE,
            root / ACTIVE,
            raw,
            milestone=milestone,
            first_id=8360,
            count=14,
        )
    )
    if tasks_digest(tasks) != result["canonical_tasks_sha256"]:
        raise ValueError("complete_task_digest")
    result.update(
        tasks=tasks, staging_disposition="existing" if staged.exists() else "consumed_by_activation"
    )
    # New tasks have no previous failure. Absence is valid only when the exact
    # frozen design also omits lineage; full-object digests still bind all bytes.
    for row, task in zip(result["contract_rows"], result["tasks"], strict=True):
        if "prior_failures" not in task:
            row["checks"]["prior"] = True
            row["matched"] = all(row["checks"].values())
            row["absolute_metric"] = int(row["matched"])
            row["raw_numerator"] = sum(row["checks"].values())
            row["excluded"] = not row["matched"]
        row.update(
            prompt_sha256=canonical_hash(task["prompt"]),
            lineage_sha256=canonical_hash(task.get("prior_failures", [])),
            full_task_sha256=canonical_hash(task),
        )
    if all(row["matched"] for row in result["contract_rows"]):
        result["gate_check_summary"] = [
            g
            for g in result["gate_check_summary"]
            if g["artifact_field"] != "contract_rows.matched"
        ]
    result["activated"] = not result["gate_check_summary"]
    return result


def disposition(operand: Json, raw: Path, refs: list[Json], failures: list[Json]) -> Json:
    """Old failed results are authenticated evidence; they are never upgraded."""
    path = Path(operand["path"])
    inputs.bind(operand, raw, refs, parse=False)
    if operand["pre_gate"]:
        value = json.loads(path.read_bytes())
        if value.get("blocked_at_layer") != "conductor_pre_gate" or value.get("experiment") != 8354:
            raise ValueError("pre_gate_identity")
        for gate in value["gates_evaluated"]:
            require_reference(dict(path=gate["artifact_path"], sha256=gate["artifact_sha256"]))
        kind = "blocked"
    else:
        value = legacy.authenticate(path, raw, refs, failures)
        kind = value.get("verdict_class")
    return dict(
        experiment_id=operand["experiment_id"],
        primary_reference=operand,
        pre_gate=operand["pre_gate"],
        honest_verdict=value.get("honest_verdict"),
        verdict_class=kind,
        required_checks_passed=value.get("required_checks_passed"),
        flagged_adversarial=value.get("flagged_adversarial"),
        gate_check_summary=value.get("gate_check_summary"),
        readiness_imported=False,
    )


def reusable(operand: Json, field: str, raw: Path, refs: list[Json]) -> Json:
    """Read the successful producer directly, rather than trusting a failed capstone."""
    problems: list[Json] = []
    value = legacy.authenticate(Path(operand["path"]), raw, refs, problems)
    if problems:
        raise ValueError(str(problems))
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    report = read_bound_sidecar(
        Path(operand["path"]), Path(terminal["publication"]["sidecar_path"])
    )
    if (
        report["report"]["passed"] is not True
        or value.get(field) != 1
        or value.get("required_checks_passed") is not True
        or value.get("flagged_adversarial") is not False
        or value.get("verdict_class") in ("blocked", "disqualified")
    ):
        raise ValueError("qualified_terminal_" + field)
    for ref in [
        value["measurement_reference"],
        *value["raw_shard_hashes"],
        *value["code_config_hashes"],
    ]:
        inputs.bind(ref, raw, refs, parse=False)
    return dict(
        ready=True,
        reference=operand,
        readiness_field=field,
        measurement_reference=value["measurement_reference"],
        honest_verdict=value["honest_verdict"],
    )


def measure(root: Path, raw: Path) -> Json:
    """Seal authenticated inputs before deriving contract or reusable-input readiness."""
    began = time.monotonic_ns()
    refs: list[Json] = []
    failures: list[Json] = []
    progress("authority_before")
    try:
        contract = authority(root, raw / "authority")
        failures.extend(contract["gate_check_summary"])
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        contract = dict(activated=False, contract_rows=[], tasks=[], canonical_tasks_sha256=None)
        failures.append(
            failure(
                root / DESIGN,
                "authority_available",
                True,
                str(error) if (root / DESIGN).exists() else None,
            )
        )
    for path in [
        DESIGN,
        STAGED,
        ACTIVE,
        "openspec/change-proposals/research-roadmap-v720-preserved-20261009.md",
        inputs.heads.PROTOCOL,
    ]:
        refs.append(snapshot(root / path, raw / "authority", str(len(refs))))
    progress("authority_after")
    frozen = inputs.load(root, raw)
    refs.extend(frozen["refs"])
    failures.extend(frozen["failures"])
    methods: Json = {}
    deployment: Json = {}
    prior: list[Json] = []
    reuse: Json = dict(kernel={}, trajectory={})
    try:
        methods = inputs.bind(dict(path=str(root / METHODS), sha256=METHODS_PIN), raw, refs)
        deployment = inputs.bind(dict(path=str(root / PROTOCOL), sha256=DEPLOYMENT_PIN), raw, refs)
        for ref in [
            methods["reference_scan"],
            methods["reference_ledger"],
            methods["ingestion_note"],
            methods["scientific_protocol"],
            methods["preserved_v720_design"],
            *methods["papers"],
            deployment["panels"]["vectors"],
            deployment["panels"]["metadata"],
            deployment["table"]["original_table"],
        ]:
            inputs.bind(ref, raw, refs, parse=False)
        for operand in methods["historical_consumers"]:
            progress("historical_before", len(prior), 14 - len(prior))
            prior.append(disposition(operand, raw, refs, failures))
            key = str(operand["experiment_id"])
            if key in methods["required_reusable"]:
                phase = "kernel" if key == "8347" else "trajectory"
                try:
                    reuse[phase] = reusable(operand, methods["required_reusable"][key], raw, refs)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    failures.append(
                        failure(
                            Path(operand["path"]), phase + "_authenticated_reuse", True, str(error)
                        )
                    )
            progress("historical_after", len(prior), 14 - len(prior))
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            error.gate
            if isinstance(error, inputs.CustodyError)
            else failure(root / METHODS, "ingested_methods", True, str(error))
        )
    atomic_json(raw / "input_manifest.json", dict(refs=refs, future_outputs_read=False))
    progress("inputs_sealed")
    return dict(
        root=str(root),
        contract=contract,
        inputs=frozen,
        protocol=frozen["protocol"],
        protocol_sha256=frozen["protocol_sha256"],
        deployment=deployment,
        methods=methods,
        support=dict(ready=bool(frozen["support"]), counts=frozen["support"].get("counts", {})),
        historical_model_provenance=frozen["historical"],
        refs=refs,
        failures=failures,
        input_failures=list(failures),
        historical_dispositions=prior,
        history={},
        numeric={},
        **reuse,
        started_monotonic_ns=began,
        ended_monotonic_ns=time.monotonic_ns(),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Authority, head and trajectory readiness are separate engineering conclusions."""
    value = dict(legacy.build(work, receipts, raw, output))
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    kind = "disqualified" if not owned else "blocked" if work["failures"] else "circular_positive"
    execution_ref = work.get("execution_manifest_reference")
    value.update(
        experiment_id=8360,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261010",
        honest_verdict="complete_" + kind + "_contract_methods",
        verdict_class=kind,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        current_contract_ready_score=int(owned and work["contract"]["activated"]),
        frozen_heads_ready_score=int(owned and bool(work["inputs"]["heads"])),
        frozen_trajectory_ready_score=int(owned and work["trajectory"].get("ready", False)),
        frozen_predictions_ready_score=int(owned and bool(work["inputs"]["predictions"])),
        local_kernel_ready_score=int(owned and work["kernel"].get("ready", False)),
        protocol_path=str(Path(work["root"]) / PROTOCOL),
        protocol_sha256=DEPLOYMENT_PIN if work["deployment"] else None,
        scientific_protocol_sha256=work["protocol_sha256"],
        deployment_protocol=work["deployment"],
        historical_dispositions=work["historical_dispositions"],
        input_manifest_reference=dict(
            path=str(raw / "input_manifest.json"), sha256=sha256_file(raw / "input_manifest.json")
        )
        if (raw / "input_manifest.json").exists()
        else None,
        execution_manifest=json.loads(Path(execution_ref["path"]).read_bytes())
        if execution_ref
        else {},
        random_seed=7218360,
        methodology_note="Administrative full-task agreement uses reference oracles. Authenticate original heads, static seals, local kernel and delayed trajectory independently; preserve all V720 dispositions. Freeze decision-preserving deployment methods before measurements. No decision benefit, current LLM inference or generator updates are measured; H1/H2 are already exposed.",
    )
    value["acceptance_gates"].update(
        frozen_heads=bool(value["frozen_heads_ready_score"]),
        frozen_trajectory=bool(value["frozen_trajectory_ready_score"]),
        local_kernel=bool(value["local_kernel_ready_score"]),
        frozen_protocol=bool(work["deployment"]),
    )
    value["gate_check_summary"] = [
        *work["failures"],
        *[
            dict(
                upstream="owned_validation",
                artifact_path=r.get("stdout_path"),
                artifact_hash=r.get("stdout_sha256"),
                artifact_field=r.get("name", "passed"),
                op="==",
                expected=True,
                observed=False,
                passed=False,
            )
            for r in receipts
            if not r["passed"]
        ],
    ]
    value["field_principles"].update(
        {
            k: "Bind frozen deployment scope to exact authenticated inputs; preserve historical outcomes without science credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        frozen_heads_ready_score="Qualified original heads remain reusable independently of contract and unrelated historical failures.",
        frozen_trajectory_ready_score="The original delayed-learning trajectory seal grants custody, not demonstrated learning benefit.",
        deployment_protocol="Freeze actions, geometry, arithmetic, panels, durable transaction scope and engineering gates before measurement.",
        historical_dispositions="Preserve thirteen producer primaries and the conductor pre-gate receipt; never repair old qualification.",
        input_manifest_reference="Seal exact input hashes before any new deployment measurements; future task outputs are not inputs.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Recompute primitive meaning; self-consistent report hashes are insufficient."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
        ]:
            require_reference(ref)
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        for key in [
            "execution_manifest_reference",
            "owned_coverage_reference",
            "input_manifest_reference",
        ]:
            if value.get(key):
                require_reference(value[key])
        for receipt in value["validation_receipts"]:
            recorded = Path(receipt["stdout_path"]).with_suffix(".receipt.json")
            if json.loads(recorded.read_bytes()) != receipt:
                return False
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        with TemporaryDirectory(prefix="exp8360-cold-") as directory:
            private = Path(directory)
            for name, ref in zip([DESIGN, STAGED, ACTIVE], work["refs"][:3], strict=True):
                if ref["exists"]:
                    p = private / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(Path(ref["snapshot_path"]).read_bytes())
            original = authority
            with patch(__name__ + ".authority", lambda root, raw: original(private, raw)):
                actual = measure(Path(work["root"]), private / "assessment")
            for key in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]:
                if actual["contract"][key] != work["contract"][key]:
                    return False
            for key in [
                "inputs",
                "deployment",
                "methods",
                "support",
                "historical_dispositions",
                "kernel",
                "trajectory",
                "protocol",
                "protocol_sha256",
                "historical_model_provenance",
                "input_failures",
            ]:
                left, right = actual[key], work[key]
                if key == "inputs":
                    left, right = (
                        {k: v for k, v in obj.items() if k != "refs"} for obj in [left, right]
                    )
                if key == "input_failures":
                    left = [
                        dict(
                            g,
                            artifact_path=str(
                                Path(work["root"]) / Path(g["artifact_path"]).relative_to(private)
                            ),
                        )
                        if Path(g["artifact_path"]).is_relative_to(private)
                        else g
                        for g in left
                    ]
                if left != right:
                    return False
        if value["source_artifact_hashes"] != work["refs"]:
            return False
        return bool(
            build(
                work,
                value["validation_receipts"],
                Path(value["work_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
