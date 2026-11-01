"""REQ-REPORT-8346: exact authority and independently reusable frozen science."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v717_contract_methods as base
from carnot.reporting import v718_contract_replay as legacy
from carnot.reporting import v718_replay_history as history
from carnot.reporting import v720_frozen_inputs as inputs
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT, DESIGN, ACTIVE, STAGED, PROTOCOL = (
    legacy.ROOT,
    legacy.DESIGN,
    legacy.ACTIVE,
    legacy.STAGED,
    legacy.PROTOCOL,
)
NAME, TASK, MILESTONE = (
    "experiment_8346_v720_frozen_input_contract",
    "exp8346-frozen-input-contract",
    "2026.10.720",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v720_frozen_input_contract_8346.py"
METHODS = "openspec/change-proposals/v720-methods-manifest.json"
METHODS_PIN = "sha256:2799390392e72108d784985e48db0bc1cc323e0b07447f194cf9a2825a2f9fca"
OWNED = [
    "python/carnot/reporting/v720_frozen_inputs.py",
    "python/carnot/reporting/v720_frozen_input_contract.py",
    "python/carnot/reporting/v720_frozen_input_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
failure = base.failure


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose real work without a synthetic elapsed-time floor."""
    print(f"[exp8346] phase={phase} completed={completed} pending={pending}", flush=True)


def design(root: Path, milestone: str) -> Path:
    """Historical readers select preserved designs, never the current live roadmap."""
    paths: dict[str, str] = {
        "2026.10.716": history.OLD_DESIGN,
        "2026.10.717": history.PRIOR_DESIGN,
        "2026.10.718": "openspec/change-proposals/research-roadmap-v718-preserved-20261009.md",
        "2026.10.719": "openspec/change-proposals/research-roadmap-v719-preserved-20261009.md",
        MILESTONE: DESIGN,
    }
    return root / paths[milestone]


def authority(root: Path, raw: Path, milestone: str = MILESTONE) -> Json:
    """Reuse the complete-object reader while explicitly selecting immutable history."""
    with patch.object(history, "design", design):
        return dict(legacy.authority(root, raw, milestone))


def measure(root: Path, raw: Path) -> Json:
    """Current authority, frozen inputs and literature have independent byte receipts."""
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
    for path in [DESIGN, STAGED, ACTIVE, str(design(Path(), "2026.10.719")), PROTOCOL]:
        refs.append(snapshot(root / path, raw / "authority", str(len(refs))))
    progress("authority_after")
    frozen = inputs.load(root, raw)
    refs.extend(frozen["refs"])
    failures.extend(frozen["failures"])
    methods: Json = {}
    prior: list[Json] = []
    try:
        methods = inputs.bind(dict(path=str(root / METHODS), sha256=METHODS_PIN), raw, refs)
        for ref in [methods["reference_scan"], methods["ingestion_note"], *methods["papers"]]:
            inputs.bind(ref, raw, refs, parse=False)
        for operand in methods["historical_consumers"]:
            progress(
                "historical_consumer_before",
                len(prior),
                len(methods["historical_consumers"]) - len(prior),
            )
            problems: list[Json] = []
            inputs.bind(operand["primary_reference"], raw, refs, parse=False)
            value = legacy.authenticate(
                Path(operand["primary_reference"]["path"]), raw, refs, problems
            )
            inputs.bind(operand["operand_reference"], raw, refs, parse=False)
            prior.append(
                dict(
                    experiment_id=operand["experiment_id"],
                    milestone=operand["milestone"],
                    operand_root=operand["operand_root"],
                    authenticated=bool(value) and not problems,
                    verdict_class=value.get("verdict_class"),
                    honest_verdict=value.get("honest_verdict"),
                    authentication_failures=problems,
                    readiness_imported=False,
                )
            )
            failures.extend(problems)
            progress(
                "historical_consumer_after",
                len(prior),
                len(methods["historical_consumers"]) - len(prior),
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            error.gate
            if isinstance(error, inputs.CustodyError)
            else failure(root / METHODS, "ingested_methods", True, str(error))
        )
    progress("methods_bound")
    return dict(
        root=str(root),
        contract=contract,
        inputs=frozen,
        protocol=frozen["protocol"],
        protocol_sha256=frozen["protocol_sha256"],
        support=dict(ready=bool(frozen["support"]), counts=frozen["support"].get("counts", {})),
        historical_model_provenance=frozen["historical"],
        refs=refs,
        failures=failures,
        methods=methods,
        prior_dispositions=prior,
        history={},
        numeric={},
        started_monotonic_ns=began,
        ended_monotonic_ns=time.monotonic_ns(),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Failed history cannot erase authentic heads; failed owned checks clear readiness."""
    value = dict(legacy.build(work, receipts, raw, output))
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    contract = work["contract"]
    current = (
        owned
        and contract["activated"]
        and len(contract["contract_rows"]) == 14
        and all(r["matched"] for r in contract["contract_rows"])
    )
    head, pred = work["inputs"]["heads"], work["inputs"]["predictions"]
    manifest_ref = work.get("execution_manifest_reference")
    execution = json.loads(Path(manifest_ref["path"]).read_bytes()) if manifest_ref else {}
    kind = "disqualified" if not owned else "blocked" if work["failures"] else "circular_positive"
    value.update(
        experiment_id=8346,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        honest_verdict="complete_" + kind + "_frozen_input_contract",
        verdict_class=kind,
        required_checks_passed=owned,
        current_contract_ready_score=int(current),
        frozen_heads_ready_score=int(owned and bool(head)),
        frozen_predictions_ready_score=int(owned and bool(pred)),
        cached_support_ready_score=int(owned and work["support"]["ready"]),
        random_seed=7208346,
        input_receipts=dict(
            reserved_labels_parsed=0,
            checkpoint=head.get("manifest", {}).get("checkpoint_reference"),
            prediction_manifest=pred.get("reference"),
            historical_failures_grant_readiness=False,
            reserved_usable_totals="authenticated_8305_summary; only public features recounted",
        ),
        role_counts=work["support"]["counts"],
        execution_manifest=execution,
        frozen_policy=dict(
            checkpoint=head.get("manifest", {}).get("checkpoint_reference"),
            whole_model_sha256=head.get("manifest", {}).get("whole_model_sha256"),
            selected_comparator=head.get("manifest", {}).get("selected_comparator"),
            temperatures={
                h["arm"]: h["temperature"] for h in head.get("manifest", {}).get("heads", [])
            },
            action_rules=work["protocol"].get("policy"),
            optimizer_config=head.get("optimizer_config"),
            role_hashes=head.get("manifest", {}).get("role_hashes"),
        ),
        historical_consumers=work["methods"].get("historical_consumers", []),
        history_reader_ready_score=0,
        historical_dispositions=work["prior_dispositions"],
        methodology_note="Authenticate original heads and sealed decisions without refitting, generation or reserved label parsing. Oracle administrative agreement is circular_positive. Reserved usable totals are imported, not reevaluated. H1/H2 remain unmeasured; capacity and lookup studies are separate dependencies.",
    )
    value["acceptance_gates"].update(
        owned_checks=owned,
        current_authority=current,
        frozen_heads=bool(value["frozen_heads_ready_score"]),
        frozen_predictions=bool(value["frozen_predictions_ready_score"]),
    )
    value["field_principles"].update(
        {
            "input_receipts": "Authenticate immutable checkpoint and sealed decisions without reserved target parsing or new inference.",
            "role_counts": "Retain intended, feature, transport and usable counts; reserved target counts are authenticated historical summaries.",
            "frozen_policy": "Freeze the exact original checkpoint, comparator, temperatures, objective, role hashes and conservative actions.",
            "historical_consumers": "Supply explicit immutable milestone and operand roots; never swap live roadmap bytes.",
            "frozen_heads_ready_score": "Authenticated heads qualify independently of current authority and historical failures, subject to owned checks.",
            "frozen_predictions_ready_score": "Authenticated original prediction seals qualify independently of historical failures, subject to owned checks.",
            "execution_manifest": "Freeze validation argv, bounded deadlines and private roots before measurement.",
            "history_reader_ready_score": "This invocation imports no historical science readiness and does not qualify unrelated branch replay.",
        }
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Recompute custody from pinned bytes; rehashing reported primitives grants nothing."""
    try:
        value = json.loads(path.read_bytes())
        for ref in [
            value["work_reference"],
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
        ]:
            require_reference(ref)
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if work.get(field):
                require_reference(work[field])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        with TemporaryDirectory(prefix="exp8346-cold-") as directory:
            private = Path(directory)
            for name, ref in zip(
                [DESIGN, STAGED, ACTIVE, str(design(Path(), "2026.10.719")), PROTOCOL],
                work["refs"][:5],
                strict=True,
            ):
                if ref["exists"]:
                    target = private / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
            try:
                checked = authority(private, private / "assessment")
            except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError):
                checked = dict(
                    activated=False, contract_rows=[], tasks=[], canonical_tasks_sha256=None
                )
            if any(
                checked[k] != work["contract"][k]
                for k in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]
            ):
                return False
            actual = inputs.load(Path(work["root"]), private / "reuse")
            if {k: v for k, v in actual.items() if k != "refs"} != {
                k: v for k, v in work["inputs"].items() if k != "refs"
            }:
                return False
            if (
                work["support"]
                != dict(ready=bool(actual["support"]), counts=actual["support"].get("counts", {}))
                or work["protocol"] != actual["protocol"]
                or work["protocol_sha256"] != actual["protocol_sha256"]
                or work["historical_model_provenance"] != actual["historical"]
            ):
                return False
            methods_ref = next(r for r in work["refs"] if r["path"].endswith(METHODS))
            if methods_ref["exists"] and methods_ref["sha256"] != METHODS_PIN:
                return False
            methods = (
                json.loads(Path(methods_ref["snapshot_path"]).read_bytes())
                if methods_ref["exists"]
                else {}
            )
            if methods != work["methods"] or work["refs"] != value["source_artifact_hashes"]:
                return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
