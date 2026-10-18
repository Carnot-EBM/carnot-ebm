"""REQ-REPORT-8318: distinguish authority, historical replay and cached support."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from carnot.reporting import v717_contract_methods as base
from carnot.reporting import v717_capstone_evidence as old
from carnot.reporting import v718_replay_history as h
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.primary_publication import validate_primary
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8318_v718_contract_replay"
TASK, MILESTONE = "exp8318-contract-replay", "2026.10.718"
DESIGN, ACTIVE, STAGED, PROTOCOL = base.DESIGN, base.ACTIVE, base.STAGED, base.PROTOCOL
METHODS = "openspec/change-proposals/v718-methods-manifest.json"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_v718_contract_replay_8318.py"
OWNED = [
    "python/carnot/reporting/v718_contract_replay.py",
    "python/carnot/reporting/v718_replay_history.py",
    "python/carnot/reporting/v718_replay_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
failure = base.failure


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose real work without artificial runtime padding."""
    print(f"[exp8318] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path, raw: Path, milestone: str = MILESTONE) -> Json:
    """Complete prompts and activated bytes must agree, even after staging is consumed."""
    path = h.design(root, milestone)
    text = path.read_text()
    tasks = parse_design(text, milestone=milestone)[1]
    digest = canonical_hash(tasks)[7:]
    declared = re.search(r"Canonical (?:tasks SHA-256|task digest): `([a-f0-9]{64})`", text)
    if declared is None or declared[1] != digest:
        raise ValueError("complete_task_digest")
    raw.mkdir(parents=True, exist_ok=True)
    reader = raw / "reader.md"
    reader.write_text(text + "\nCanonical full-task SHA256: `" + digest + "`\n")
    staged = root / STAGED
    result = assess_authorities(
        reader,
        staged if staged.exists() else root / ACTIVE,
        root / ACTIVE,
        raw / "assessment",
        milestone=milestone,
        first_id=int(tasks[0]["id"][3:7]),
        count=14,
    )
    result["tasks"] = tasks
    result["staging_disposition"] = "existing" if staged.exists() else "consumed_by_activation"
    return dict(result)


def authenticate(path: Path, raw: Path, refs: list[Json], failures: list[Json]) -> Json:
    """Authentication permits reading failures but never upgrades their verdict."""
    before = len(failures)
    value = base.bind(path, raw, refs, failures, terminal=True)
    if len(failures) != before:
        return {}
    try:
        validate_primary(value, path)
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        if terminal["publication"]["primary_sha256"] != sha256_file(path):
            raise ValueError("terminal_primary_sha256")
        for receipt in terminal.get("checks", []):
            for stream in ["stdout", "stderr"]:
                ref = snapshot(Path(receipt[stream + "_path"]), raw / "logs", str(len(refs)))
                require_reference(dict(ref, sha256=receipt[stream + "_sha256"]))
                refs.append(ref)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(failure(path, "authenticated_terminal_bytes", True, str(error)))
        return {}
    return dict(value)


def measure(root: Path, raw: Path) -> Json:
    """Only declared historical operands are read; future producers are dependencies."""
    start = time.monotonic_ns()
    refs: list[Json] = []
    failures: list[Json] = []
    progress("authority_before")
    try:
        contract = authority(root, raw / "authority")
        failures.extend(contract["gate_check_summary"])
    except (OSError, ValueError, KeyError, IndexError, TypeError, yaml.YAMLError) as error:
        contract = dict(activated=False, contract_rows=[], tasks=[], canonical_tasks_sha256=None)
        failures.append(failure(root / DESIGN, "authority_available", True, str(error)))
    for path in [DESIGN, STAGED, ACTIVE, h.OLD_DESIGN, h.PRIOR_DESIGN]:
        refs.append(snapshot(root / path, raw / "authority", Path(path).stem))
    protocol = base.bind(root / PROTOCOL, raw, refs, failures)
    digest = sha256_file(root / PROTOCOL) if (root / PROTOCOL).exists() else None
    if digest != base.PIN:
        failures.append(failure(root / PROTOCOL, "protocol_sha256", base.PIN, digest))
    methods = base.bind(root / METHODS, raw, refs, failures)
    for operand in [
        methods.get("capacity_protocol"),
        methods.get("reference_scan"),
        *methods.get("papers", []),
    ]:
        if operand:
            ref = snapshot(Path(operand["path"]), raw / "methods", str(len(refs)))
            require_reference(dict(ref, sha256=operand["sha256"]))
            refs.append(ref)
    progress("authority_after")
    positive = [
        authenticate(root / p, raw, refs, failures)
        for p in ["results/experiment_8304_v717_contract_methods.json", h.CUSTODY]
    ]
    support: Json = dict(ready=False, counts={})
    support_failures = len(failures)
    for value, field in zip(
        positive, ["protocol_ready_score", "fit_support_ready_score"], strict=True
    ):
        for key, expected in [
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
            (field, 1),
        ]:
            if value.get(key) != expected:
                failures.append(
                    failure(
                        root
                        / (
                            h.CUSTODY
                            if field == "fit_support_ready_score"
                            else "results/experiment_8304_v717_contract_methods.json"
                        ),
                        key,
                        expected,
                        value.get(key),
                    )
                )
    if all(positive) and len(failures) == support_failures and digest == base.PIN:
        support = h.support(positive[1], raw, refs)
    history: Json = {}
    numeric: Json = {}
    progress("history_before")
    try:
        authenticate(root / "results/experiment_8317_v717_capstone.json", raw, refs, failures)
        history = h.historical(root, raw)
        numeric = h.numeric(root, raw, refs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            failure(
                root / "results/experiment_8317_v717_capstone.json",
                "historical_primitives",
                True,
                str(error),
            )
        )
    progress("history_after")
    health_path = root / "results/raw/experiment_8318_v718_contract_replay/global_health.json"
    health = base.bind(health_path, raw, refs, failures) if health_path.is_file() else {}
    return dict(
        root=str(root),
        contract=contract,
        protocol=protocol,
        protocol_sha256=digest,
        methods=methods,
        refs=refs,
        failures=failures,
        support=support,
        history=history,
        numeric=numeric,
        global_health=health,
        historical_model_provenance=positive[1].get("historical_model_provenance", []),
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Frozen pre-reduction checks qualify three independent execution gates."""
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    unknown = bool(work["history"] and not work["history"]["deterministic"])
    history_ready = owned and not unknown and work["history"].get("deterministic", False)
    current = owned and not unknown and work["contract"]["activated"]
    cached = owned and not unknown and work["support"]["ready"]
    kind = (
        "disqualified"
        if unknown or any(not r["passed"] for r in receipts) or (not owned and not work["failures"])
        else "blocked"
        if work["failures"]
        else "circular_positive"
    )
    rows = [
        dict(
            row,
            completed=True,
            status="completed",
            failed=not row["matched"],
            excluded=not row["matched"],
            numerator=row["absolute_metric"],
            denominator=1,
        )
        for row in work["contract"]["contract_rows"]
    ]
    value = dict(
        experiment_id=8318,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261008",
        honest_verdict="complete_"
        + kind
        + "_"
        + (work["failures"][0]["artifact_field"] if kind == "blocked" else "contract_replay"),
        verdict_class=kind,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        no_model_load=True,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical_model_provenance"],
        rows=rows,
        intended_count=14,
        completed_count=len(rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=14 - len(rows),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        sample_size_budget=dict(
            administrative_tasks=14, independent_natural_observations=0, H1=128, H2=88
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned and not unknown,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_checks=owned,
            current_authority=current,
            history_reader=history_ready,
            cached_support=cached,
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("finding_audits", []),
        finding_consumer_policy=h.POLICY,
        finding_dispositions=[a["dispositions"] for a in work.get("finding_audits", [])],
        policy_negative_controls=work.get("policy_controls", []),
        preconditions_checked=work.get("preconditions", {}),
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="aggregation",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
            )
        ],
        random_seed=7188318,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work.get("code_refs", []),
        raw_shard_hashes=[
            dict(path=str(raw / "measurement.json"), sha256=sha256_file(raw / "measurement.json"))
        ],
        cited_upstream_artifacts=[
            dict(r, fields_imported=["byte custody and historical disposition"])
            for r in work["refs"]
        ],
        current_contract_ready_score=int(current),
        history_reader_ready_score=int(history_ready),
        cached_support_ready_score=int(cached),
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        protocol_path=str(Path(work["root"]) / PROTOCOL),
        protocol_sha256=work.get("protocol_sha256"),
        methods_manifest=work["methods"],
        authority_snapshots=work["refs"][:5],
        first_reduction_mismatch=work["history"].get("first_reduction_mismatch"),
        historical_dispositions=work["history"].get("recomputed", {}),
        historical_primary_verdict=work["history"].get("historical_honest_verdict"),
        historical_reduction_sha256=work["history"].get("reduction_sha256"),
        cached_support=work["support"],
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        invocation_argv=work.get("invocation_argv", []),
        future_dependencies=[
            dict(task_id=t["id"], path=t["deliverable"], status="dependency_not_input")
            for t in work["contract"]["tasks"][1:]
        ],
        work_reference=dict(
            path=str(raw / "measurement.json"), sha256=sha256_file(raw / "measurement.json")
        ),
        publication_output=str(output),
        methodology_note="Exact administrative authority is constructed oracle agreement. Stable serialized historical reduction preserves failed science and self slots. Imported cached development supplies no independent generalization.",
        global_health=work.get("global_health", {}),
    )
    value["field_principles"] = {
        k: "Bind this conclusion to byte-bound primitives; execution readiness never upgrades failed or unmeasured science."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    purposes = {
        "experiment_id task_id milestone run_date invocation_argv publication_output": "Identify the exact invocation and its activated execution authority.",
        "honest_verdict verdict_class gate_check_summary required_checks_passed flagged_adversarial acceptance_gates": "Separate administrative oracle agreement, missing external operands and failed owned qualification.",
        "inference_substrate inference_substrate_class MODEL_SPECS no_model_load model_invocation_counts historical_model_provenance": "Count zero current model work; imported Qwen receipts retain only historical credit.",
        "rows intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget future_dependencies": "Retain fourteen intended administrative units; future paths and repetitions are not observations.",
        "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score": "Constructed oracle agreement and exposed development give no independent generalization credit.",
        "validation_receipts terminal_validation_sidecar_path execution_manifest_reference owned_coverage_reference": "Bind readiness to exact frozen commands and byte-bound checks; terminal receipts stay outside reduction.",
        "preconditions_checked duration_s phase_spans random_seed reproducibility_checksum source_artifact_hashes code_config_hashes raw_shard_hashes cited_upstream_artifacts": "Bind conclusions to authenticated operands, real clocks, copied source bytes and reproducible code.",
        "current_contract_ready_score canonical_tasks_sha256 authority_snapshots": "Require exact fourteen-task full-object and table equality with conductor-activated authority.",
        "history_reader_ready_score first_reduction_mismatch historical_dispositions historical_primary_verdict historical_reduction_sha256": "Require deterministic private cold replay of all fields while preserving old disqualification and absent H1/H2.",
        "cached_support_ready_score cached_support": "Recount original fit/tune roles and class support from separate authenticated predictor/evaluator shards.",
        "protocol_path protocol_sha256 methods_manifest": "Bind unchanged V717 science and a separate constructed capacity protocol to full primary paper methods.",
        "adversarial_findings finding_consumer_policy finding_dispositions policy_negative_controls": "Preserve findings; resolve known info only by independent arithmetic and deliberate error rejection.",
        "global_health": "Retain the separately requested bounded full-suite failure; do not report all repository tests as passing.",
    }
    value["field_principles"].update(
        {field: purpose for fields, purpose in purposes.items() for field in fields.split()}
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rehashed summaries must still equal every field derived from frozen primitives."""
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
        if not replay_operands(work) or work["refs"] != value["source_artifact_hashes"]:
            return False
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                require_reference(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        if work["history"]:
            history = work["history"]
            require_reference(history["original_candidate"])
            require_reference(history["work_reference"])
            primitive = json.loads(Path(history["work_reference"]["snapshot_path"]).read_bytes())
            actual = h.restore_locations(
                old.reduce(primitive, history["historical_receipts"]), history["relocation_map"]
            )
            if (
                actual != history["recomputed"]
                or canonical_hash(actual) != history["reduction_sha256"]
            ):
                return False
            original = json.loads(Path(history["original_candidate"]["snapshot_path"]).read_bytes())
            if (
                h.first_difference({k: original[k] for k in actual}, actual)
                != history["first_reduction_mismatch"]
            ):
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


def replay_operands(work: Json) -> bool:
    """Rebuild authority and support from copied operands rather than reported gates."""
    with TemporaryDirectory(prefix="exp8318-cold-") as directory:
        private = Path(directory)
        for name, ref in zip(
            [DESIGN, STAGED, ACTIVE, h.OLD_DESIGN, h.PRIOR_DESIGN], work["refs"][:5], strict=True
        ):
            if ref["exists"]:
                target = private / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(ref["snapshot_path"]).read_bytes())
        try:
            actual = authority(private, private / "assessment")
        except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError):
            actual = dict(activated=False, contract_rows=[], tasks=[], canonical_tasks_sha256=None)
        if any(
            actual[k] != work["contract"][k]
            for k in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]
        ):
            return False
        source_ref = next(r for r in work["refs"] if r["path"].endswith(h.CUSTODY))
        if source_ref["exists"] and work["support"].get("counts"):
            source = json.loads(Path(source_ref["snapshot_path"]).read_bytes())
            for field in ["predictor_shards", "evaluator_shards"]:
                for operand in source[field].values():
                    bound = next(r for r in work["refs"] if r["path"] == operand["path"])
                    operand["path"] = bound["snapshot_path"]
            recounted = h.support(source, private / "support", [])
            if any(
                recounted[k] != work["support"][k] for k in ["ready", "counts", "manifest_sha256"]
            ):
                return False
        protocol_ref = next(r for r in work["refs"] if r["path"].endswith(PROTOCOL))
        protocol = (
            json.loads(Path(protocol_ref["snapshot_path"]).read_bytes())
            if protocol_ref["exists"]
            else {}
        )
        return bool(protocol == work["protocol"])
