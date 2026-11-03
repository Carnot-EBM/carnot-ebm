"""REQ-REPORT-8304: freeze administrative evidence without claiming natural benefit."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import re
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.v710_contract_replay import require_reference, snapshot

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8304_v717_contract_methods"
TASK = "exp8304-contract-methods"
MILESTONE = "2026.10.717"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
PROTOCOL = "openspec/change-proposals/v717-local-learning-protocol.json"
PIN = "sha256:853709123024de763e96dd688e819f0430205ae6d97d6561a2b95cca23b81c6f"
ACTIVE, STAGED = "research-roadmap.yaml", "research-roadmap-next.yaml"
HISTORY = "results/experiment_8303_v716_capstone.json"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_v717_contract_methods_8304.py"
OWNED = [
    "python/carnot/reporting/v717_contract_methods.py",
    "python/carnot/reporting/v717_contract_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so the conductor can distinguish work from a stalled child."""
    print(f"[exp8304] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Keep missing operands null rather than inventing a measured zero."""
    return dict(
        upstream=path.stem,
        artifact_path=str(path),
        artifact_hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def bind(
    path: Path, raw: Path, refs: list[Json], failures: list[Json], terminal: bool = False
) -> Json:
    """Copy authenticated bytes before importing a historical disposition."""
    ref = snapshot(path, raw / "inputs", path.stem)
    refs.append(
        dict(
            ref, fields_imported=["terminal disposition"] if terminal else ["immutable input bytes"]
        )
    )
    operand = path
    try:
        value: Json = json.loads(path.read_bytes())
        if terminal:
            declared = value.get("terminal_validation_sidecar_path")
            sidecar = (
                Path(declared)
                if declared
                else path.parent
                / "raw"
                / path.stem
                / "validators"
                / (sha256_file(path).split(":")[1] + ".json")
            )
            report = json.loads(sidecar.read_bytes())
            refs.append(snapshot(sidecar, raw / "inputs", "terminal"))
            if "publication" in report:
                sidecar = Path(report["publication"]["sidecar_path"])
                refs.append(snapshot(sidecar, raw / "inputs", "validator"))
            operand = sidecar
            read_bound_sidecar(path, sidecar)
        return value
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            failure(
                operand,
                "authenticated_terminal" if terminal else "readable_json",
                True,
                str(error) if operand.is_file() else None,
            )
        )
        return {}


def authority(paths: list[Path], raw: Path) -> Json:
    """Normalize only the reader's digest annotation; original bytes remain snapshotted."""
    text = paths[0].read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    declared = re.search(r"Canonical tasks SHA-256: `([0-9a-f]{64})`", text)
    if declared is None or declared.group(1) != tasks_digest(tasks):
        raise ValueError("design_complete_task_digest")
    reader = raw / "reader_design.md"
    reader.parent.mkdir(parents=True, exist_ok=True)
    reader.write_text(text + "\nCanonical full-task SHA256: `" + tasks_digest(tasks) + "`\n")
    return dict(
        assess_authorities(
            reader, *paths[1:], raw / "assessment", milestone=MILESTONE, first_id=8304, count=14
        )
    )


def measure(root: Path, raw: Path) -> Json:
    """Read protocol first; future producer files remain dependencies, not observations."""
    progress("preconditions")
    start = time.monotonic_ns()
    wall = time.time_ns()
    refs: list[Json] = []
    failures: list[Json] = []
    protocol = bind(root / PROTOCOL, raw, refs, failures)
    if refs[0]["sha256"] != PIN:
        failures.append(failure(root / PROTOCOL, "protocol_sha256", PIN, refs[0]["sha256"]))
    for item in protocol.get("method_mapping", []) + [protocol.get("source_roster_authority", {})]:
        if item:
            path = Path(item["path"])
            ref = snapshot(path, raw / "inputs", path.stem)
            refs.append(ref)
            if ref["sha256"] != item["sha256"]:
                failures.append(failure(path, "sha256", item["sha256"], ref["sha256"]))
    paths = [root / name for name in (DESIGN, STAGED, ACTIVE)]
    authorities = [
        snapshot(p, raw / "authority", role)
        for p, role in zip(paths, ["design", "staged", "active"])
    ]
    refs.extend(authorities)
    try:
        contract = authority(paths, raw / "contract")
        tasks = parse_design(paths[0].read_text(), milestone=MILESTONE)[1]
        failures.extend(dict(g, upstream="V717_authority") for g in contract["gate_check_summary"])
    except (OSError, ValueError, KeyError, IndexError, TypeError, yaml.YAMLError) as error:
        contract = dict(activated=False, contract_rows=[], canonical_tasks_sha256=None)
        tasks = []
        failures.append(failure(paths[0], "authority_readable", True, str(error)))
    contract["authority_snapshots"] = authorities
    progress("protocol_frozen_before_history")
    if root == ROOT:
        for name in [
            "experiment_7996_v693_sparse_energy_training",
            "experiment_7998_v693_selective_feedback_learning",
        ]:
            bind(root / "results" / (name + ".json"), raw, refs, failures, terminal=True)
    capstone = bind(root / HISTORY, raw, refs, failures, terminal=True)
    history = []
    for index, disposition in enumerate(capstone.get("task_dispositions", [])):
        item = dict(disposition)
        path = Path(item["path"]) if item.get("path") else root / HISTORY
        ref = snapshot(path, raw / "history", str(index))
        refs.append(ref)
        if item.get("sha256") and ref["sha256"] != item["sha256"]:
            failures.append(
                failure(path, "capstone_disposition_sha256", item["sha256"], ref["sha256"])
            )
        if item.get("producer_executed") and item.get("path"):
            source = bind(path, raw, refs, failures, terminal=True)
            item.update(
                honest_verdict=source.get("honest_verdict"),
                verdict_class=source.get("verdict_class"),
            )
        elif not ref["exists"]:
            item.pop("honest_verdict", None)
            item.pop("verdict_class", None)
        history.append(item)
        progress("history", index + 1, len(capstone["task_dispositions"]) - index - 1)
    work = dict(
        contract=contract,
        tasks=tasks,
        protocol=protocol,
        refs=refs,
        failures=failures,
        history=history,
        protocol_path=str(root / PROTOCOL),
        protocol_sha256=refs[0]["sha256"],
        started_monotonic_ns=start,
        ended_monotonic_ns=time.monotonic_ns(),
        started_wall_ns=wall,
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Administrative readiness requires checked choices, never a CUDA or benefit score."""
    atomic_json(raw / "measurement.json", work)
    owned = (
        bool(receipts)
        and all(r["passed"] and r["exit_code"] == r.get("expected_exit", 0) for r in receipts)
        and work.get("owned_validation_complete", True)
    )
    contract_ready = int(owned and work["contract"]["activated"])
    protocol_ready = int(owned and work["protocol_sha256"] == PIN and not work["failures"])
    owned_failure = any(not r["passed"] for r in receipts)
    verdict = (
        "disqualified"
        if owned_failure
        else "blocked"
        if work["failures"]
        else "circular_positive"
        if owned
        else "disqualified"
    )
    rows = [
        dict(
            row,
            completed=True,
            status="completed",
            excluded=False,
            failed=False,
            numerator=row["absolute_metric"],
            denominator=1,
        )
        for row in work["contract"]["contract_rows"]
    ]
    value = dict(
        experiment_id=8304,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261008",
        honest_verdict="complete_"
        + verdict
        + "_"
        + (work["failures"][0]["artifact_field"] if verdict == "blocked" else "contract_methods"),
        verdict_class=verdict,
        gate_check_summary=work["failures"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=dict(
            scope="imported V707 Qwen receipts only; zero current model calls",
            source_roster_authority=work["protocol"].get("source_roster_authority"),
        ),
        rows=rows,
        intended_count=14,
        completed_count=len(rows),
        failed_count=0,
        censored_count=14 - len(rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            administrative_tasks=14,
            independent_natural_observations=0,
            reserved_slots=128,
            stream_slots=96,
            retention_slots=32,
        ),
        verifier_is_oracle=True,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=owned,
            activated_contract=bool(contract_ready),
            immutable_protocol=bool(protocol_ready),
            scientific_benefit=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        preconditions_checked=dict(
            private_scratch=True, input_authentication=True, no_model_resources_required=True
        ),
        duration_s=(work["ended_monotonic_ns"] - work["started_monotonic_ns"]) / 1e9,
        phase_spans=[
            dict(
                phase="authority_protocol_history",
                started_monotonic_ns=work["started_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
            )
        ],
        random_seed=7178304,
        source_artifact_hashes=work["refs"],
        code_config_hashes=[snapshot(ROOT / p, raw / "code", str(i)) for i, p in enumerate(OWNED)],
        raw_shard_hashes=[
            dict(path=str(raw / "measurement.json"), sha256=sha256_file(raw / "measurement.json"))
        ],
        cited_upstream_artifacts=work["refs"],
        current_contract_ready_score=contract_ready,
        protocol_ready_score=protocol_ready,
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        authority_snapshots=work["contract"]["authority_snapshots"],
        protocol_path=work["protocol_path"],
        protocol_sha256=work["protocol_sha256"],
        method_mapping=work["protocol"].get("method_mapping", []),
        prior_null_differences=dict(
            Exp7426_7427="Static/online spline null; changed bounded local sentence inputs and exact recovery.",
            Exp7996_7998="Nine scalar/lexical inputs were null; four authenticated local relation inputs and fixed-intercept sparse writes differ.",
            Exp8183_8185="Global radial sentence heads were null; local fixed support and matched-information controls now isolate locality.",
        ),
        parked_v713_chain=work["protocol"].get("parked_v713_chain", {}),
        v716_dispositions=work["history"],
        upstream_dependencies=[
            dict(task_id=t["id"], path=t["deliverable"], status="dependency_not_historical")
            for t in work["tasks"][1:]
        ],
        work_reference=dict(
            path=str(raw / "measurement.json"), sha256=sha256_file(raw / "measurement.json")
        ),
        invocation_argv=work.get("invocation_argv", []),
        execution_manifest_reference=work.get("execution_manifest_reference"),
        owned_coverage_reference=work.get("owned_coverage_reference"),
        methodology_note="Fourteen exact administrative comparisons and frozen methods. Oracle agreement is structural; no science or independent generalization is measured.",
    )
    groups = {
        "experiment_id task_id milestone run_date": "Identify this invocation and execution authority.",
        "honest_verdict verdict_class gate_check_summary": "Distinguish owned disqualification, external block and administrative oracle agreement.",
        "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts historical_model_provenance": "Imported receipts never become current model execution.",
        "rows intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget": "Preserve all fourteen administrative comparisons without claiming independent natural samples.",
        "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score": "Exposed development and oracle agreement provide no independent generalization.",
        "required_checks_passed flagged_adversarial acceptance_gates validation_receipts terminal_validation_sidecar_path": "Readiness is conditional on actual owned commands and byte-bound terminal validation.",
        "current_contract_ready_score protocol_ready_score canonical_tasks_sha256 authority_snapshots protocol_path protocol_sha256": "Exact full-task authority and immutable scientific choices have separate readiness operands.",
        "method_mapping prior_null_differences parked_v713_chain": "Bounded adaptations do not resurrect failed mechanisms or unmeasured intervention science.",
    }
    principles = {
        field: principle for fields, principle in groups.items() for field in fields.split()
    }
    value["field_principles"] = {
        k: principles.get(
            k, "Bind this receipt to immutable source bytes, measured clocks and replayable work."
        )
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute full authority and primitive readiness even after an attacker rehashes JSON."""
    from tempfile import TemporaryDirectory

    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            require_reference(ref)
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        for field in ["execution_manifest_reference", "owned_coverage_reference"]:
            if value.get(field):
                require_reference(value[field])
        if (
            work["refs"] != value["source_artifact_hashes"]
            or work["protocol_sha256"] != work["refs"][0]["sha256"]
        ):
            return False
        snapshots = work["contract"]["authority_snapshots"]
        with TemporaryDirectory(prefix="exp8304-replay-") as directory:
            if snapshots[0]["exists"]:
                paths = [
                    Path(s.get("snapshot_path", Path(directory) / "absent")) for s in snapshots
                ]
                try:
                    actual = authority(paths, Path(directory))
                except (ValueError, KeyError, IndexError, TypeError, yaml.YAMLError):
                    actual = dict(activated=False, contract_rows=[], canonical_tasks_sha256=None)
                for key in ["activated", "contract_rows", "canonical_tasks_sha256"]:
                    if actual[key] != work["contract"][key]:
                        return False
                if any(
                    not any(
                        all(
                            g.get(k) == expected.get(k)
                            for k in ["artifact_field", "op", "expected", "observed"]
                        )
                        for g in work["failures"]
                    )
                    for expected in actual.get("gate_check_summary", [])
                ):
                    return False
            original_protocol = (
                json.loads(Path(work["refs"][0]["snapshot_path"]).read_bytes())
                if work["refs"][0]["exists"]
                else {}
            )
            if original_protocol != work["protocol"] or not replay_history(work, Path(directory)):
                return False
            rebuilt = build(
                work, value["validation_receipts"], Path(directory), Path(directory) / NAME
            )
        for key in [
            "experiment_id",
            "task_id",
            "milestone",
            "run_date",
            "rows",
            "honest_verdict",
            "verdict_class",
            "MODEL_SPECS",
            "model_invocation_counts",
            "required_checks_passed",
            "current_contract_ready_score",
            "protocol_ready_score",
            "gate_check_summary",
            "intended_count",
            "completed_count",
            "independent_count",
            "method_mapping",
            "v716_dispositions",
            "parked_v713_chain",
            "protocol_sha256",
        ]:
            if value[key] != rebuilt[key]:
                return False
        for receipt in value["validation_receipts"]:
            for prefix in ["stdout", "stderr"]:
                if (
                    prefix + "_path" in receipt
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def replay_history(work: Json, private: Path) -> bool:
    """Reduce historical dispositions from the capstone and primary copies, not its headline."""
    capref = next(ref for ref in work["refs"] if ref["path"].endswith(HISTORY))
    capstone = bind(Path(capref.get("snapshot_path", private / "absent")), private, [], [])
    expected = deepcopy(capstone.get("task_dispositions", []))
    for item in expected:
        path = item.get("path") or capref["path"]
        ref = next(ref for ref in work["refs"] if ref["path"] == path)
        if item.get("producer_executed") and item.get("path"):
            source = bind(Path(ref.get("snapshot_path", private / "absent")), private, [], [])
            item.update(
                honest_verdict=source.get("honest_verdict"),
                verdict_class=source.get("verdict_class"),
            )
        elif not ref["exists"]:
            item.pop("honest_verdict", None)
            item.pop("verdict_class", None)
    return bool(expected == work["history"])
