"""REQ-REPORT-8276: current execution bindings grant no scientific benefit.

The qualified runners own supervision and publication. This adapter binds
their checks to current authority while preserving the original science.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import v714_coverage_custody as prior
from carnot.reporting import v714_coverage_runner as runner
from carnot.reporting import coverage_custody_8262 as custody
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v685_authority_lifecycle import assess_authorities, tasks_digest
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import protocol_conformance_8263 as protocol

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8276_v715_current_contract_readiness"
TASK = "exp8276-current-contract-readiness"
MILESTONE = "2026.10.715"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_current_contract_readiness_8276.py"
OWNED = ["python/carnot/reporting/current_contract_readiness_8276.py", CLI]
REUSED = list(
    dict.fromkeys(
        prior.REUSED
        + prior.OWNED
        + protocol.OWNED
        + [
            protocol.TEST,
            "python/carnot/verify/evidence_view_execution_8249.py",
            "python/carnot/verify/evidence_view_kernel_8249.py",
            "python/carnot/reporting/methods_stream_execution_8111.py",
            "python/carnot/verify/calibrated_memory_trajectory_8211.py",
            "python/carnot/verify/sentence_transport_methods_8179.py",
            "python/carnot/verify/sentence_transport_8179.py",
        ]
    )
)
DESIGN, ACTIVE, STAGED = prior.DESIGN, prior.ACTIVE, prior.STAGED
PROTOCOL, PIN = prior.PROTOCOL, prior.PIN
EXECUTION = "openspec/change-proposals/v715-evidence-execution-contract.json"
MODEL_SPECS: list[Json] = []
BASE_MANIFEST = runner.manifest
BASE_BUILD = runner.build


def assess(paths: list[Path], raw: Path) -> Json:
    """Supply a computed design digest when the original omits its printed annotation.

    The derived reader input retains every original task byte. Its digest comes
    from design JSON alone, then the existing reader compares active tasks.
    """
    text = paths[0].read_text()
    tasks = parse_design(text, milestone=MILESTONE)[1]
    reader = raw / "reader_design.md"
    reader.parent.mkdir(parents=True, exist_ok=True)
    if (
        re.search(r"Canonical (?:full-task|complete-task|task) SHA-?256: `[0-9a-f]{64}`", text)
        is None
    ):
        text += "\nCanonical full-task SHA256: `" + tasks_digest(tasks) + "`\n"
    reader.write_text(text)
    return dict(
        assess_authorities(
            reader, *paths[1:3], raw / "assessment", milestone=MILESTONE, first_id=8276, count=14
        )
    )


def authority_work(root: Path, raw: Path) -> Json:
    """Read complete task bytes; activation can survive consumed staging files."""
    refs = [
        snapshot(root / name, raw / "inputs", role)
        for name, role in [
            (DESIGN, "design"),
            (STAGED, "staged"),
            (ACTIVE, "active"),
            (PROTOCOL, "science"),
        ]
    ]
    work: Json = dict(
        refs=refs,
        failures=[],
        tasks=[],
        execution_contract={},
        contract=dict(
            activated=False,
            planning_matched=False,
            canonical_tasks_sha256=None,
            contract_rows=[],
            authority_snapshots={},
            gate_check_summary=[],
        ),
    )
    try:
        paths = [Path(r.get("snapshot_path", raw / f"absent-{i}")) for i, r in enumerate(refs)]
        contract = assess(paths[:3], raw / "authority")
        _, tasks = parse_design(paths[0].read_text(), milestone=MILESTONE)
        work.update(contract=contract, tasks=tasks)
        work["failures"].extend(
            dict(g, component="contract") for g in contract["gate_check_summary"]
        )
        paths_by_task = {t["id"]: t["deliverable"] for t in tasks}
        work["execution_contract"] = dict(
            milestone=MILESTONE,
            science_protocol_path=PROTOCOL,
            science_protocol_sha256=PIN,
            canonical_tasks_sha256=tasks_digest(tasks),
            producers=[
                dict(
                    task_id=t["id"],
                    producer_path=t["deliverable"],
                    current_gate_fields=[
                        dict(g, artifact_path=paths_by_task[g["upstream"]]) for g in t["gated_on"]
                    ],
                )
                for t in tasks
            ],
            science_choices="Unchanged V713 roles, grids, thresholds, costs, seeds and fallback.",
            historical_task_numbers="provenance_only",
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        work["failures"].append(
            dict(
                prior.failure(root / DESIGN, "authority_readable", True, str(error)),
                component="contract",
            )
        )
    if refs[3]["sha256"] != PIN:
        work["failures"].append(
            dict(
                prior.failure(root / PROTOCOL, "frozen_science_sha256", PIN, refs[3]["sha256"]),
                component="shared",
            )
        )
    for ref in refs:
        ref["fields_imported"] = (
            ["full task authority"]
            if ref != refs[3]
            else ["frozen science", "role_manifest", "head_config"]
        )
    return work


def bind_terminal(root: Path, name: str, work: Json, raw: Path, component: str) -> Json:
    """Authenticate exact primary and terminal bytes before importing any fields."""
    path = root / name
    ref = snapshot(path, raw / "inputs", path.stem)
    ref["fields_imported"] = [
        "honest_verdict",
        "verdict_class",
        "required_checks_passed",
        "flagged_adversarial",
    ]
    work["refs"].append(ref)
    try:
        value = json.loads(path.read_bytes())
        terminal = Path(value["terminal_validation_sidecar_path"])
        report = json.loads(terminal.read_bytes())
        sidecar = (
            terminal if "primary_sha256" in report else Path(report["publication"]["sidecar_path"])
        )
        bound = read_bound_sidecar(path, sidecar)
        operands = [
            (path, "required_checks_passed", True, value.get("required_checks_passed")),
            (path, "flagged_adversarial", False, value.get("flagged_adversarial")),
            (sidecar, "report.passed", True, bound["report"]["passed"]),
        ]
        if terminal != sidecar:
            operands.append(
                (
                    terminal,
                    "publication.primary_sha256",
                    sha256_file(path),
                    report["publication"]["primary_sha256"],
                )
            )
        for operand, field, expected, observed in operands:
            if observed != expected:
                work["failures"].append(
                    dict(prior.failure(operand, field, expected, observed), component=component)
                )
        for p in dict.fromkeys([terminal, sidecar]):
            work["refs"].append(
                dict(
                    snapshot(p, raw / "inputs", "terminal"),
                    fields_imported=["publication", "primary_sha256", "report.passed"],
                )
            )
        return dict(value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        operand = (
            path if not path.is_file() else locals().get("sidecar", locals().get("terminal", path))
        )
        work["failures"].append(
            dict(
                prior.failure(
                    operand,
                    "authenticated_terminal",
                    True,
                    None if not operand.is_file() else str(error),
                ),
                component=component,
            )
        )
        return {}


def measure(root: Path, raw: Path) -> Json:
    """Re-execute qualified controls; historical missing science does not gate them."""
    runner.progress("current_authority_before")
    start, wall = time.monotonic_ns(), time.time_ns()
    work = authority_work(root, raw)
    work["history"] = []
    imports = {}
    for exp, suffix, component in [
        (8262, "coverage_custody", "coverage"),
        (8263, "protocol_conformance", "protocol"),
        (8264, "evidence_view_canary", "history"),
        (8275, "capstone", "history"),
    ]:
        imports[exp] = bind_terminal(
            root, f"results/experiment_{exp}_v714_{suffix}.json", work, raw, component
        )
    for item in imports[8275].get("task_dispositions", []):
        if item["experiment_id"] == 8275:
            continue
        declared = root / item["declared_path"]
        ref = snapshot(declared, raw / "history", str(item["experiment_id"]))
        work["refs"].append(
            dict(ref, fields_imported=["producer terminal fields"] if ref["exists"] else [])
        )
        source = json.loads(declared.read_bytes()) if ref["exists"] else {}
        row = dict(
            task_id=item["task_id"],
            path=str(declared),
            sha256=ref["sha256"],
            disposition="producer_terminal" if source else "cascade_skip_absent_primary",
            source_counts={
                k: source.get(k)
                for k in [
                    "intended_count",
                    "completed_count",
                    "failed_count",
                    "censored_count",
                    "excluded_count",
                ]
            },
        )
        if source:
            row.update(
                honest_verdict=source.get("honest_verdict"),
                verdict_class=source.get("verdict_class"),
            )
        row["conductor_evidence"] = {
            k: item.get(k) for k in ["path", "sha256", "evidence_type", "producer_executed"]
        }
        work["history"].append(row)
    science = (
        json.loads(Path(work["refs"][3]["snapshot_path"]).read_bytes())
        if work["refs"][3]["exists"]
        else {}
    )
    work["source_role_manifest"] = science.get("role_manifest", {})
    work["frozen_head_config"] = science.get("head_config", {})
    work["historical_canary"] = {
        k: imports[8264].get(k)
        for k in ["honest_verdict", "gate_check_summary", "model_invocation_counts"]
    }
    with patch.object(protocol, "CLI", CLI):
        runner.progress("before_current_protocol_benchmark")
        pwork = protocol.measure(root, raw / "protocol")
        runner.progress("after_current_protocol_benchmark")
    for field in ["refs", "code_config_hashes"]:
        pwork[field] = [
            dict(
                snapshot(Path(r["path"]), raw / "protocol" / "snapshots", str(i)),
                path=str(raw / "protocol" / "snapshots" / f"unused-{i}"),
                original_path=r["path"],
            )
            for i, r in enumerate(pwork[field])
        ]
        for ref in pwork[field]:
            ref["path"] = ref.get("snapshot_path", ref["original_path"])
    atomic_json(raw / "protocol" / "measurement.json", pwork)
    work["protocol_work"] = pwork
    work["failures"].extend(
        dict(g, component="view" if g["upstream"] == "Qwen3.8_GGUF" else "protocol")
        for g in pwork["checks"]
        if not g["passed"]
    )
    work["component_imports"] = {
        str(exp): {
            k: v.get(k)
            for k in [
                "coverage_custody_ready_score",
                "view_kernel_ready_score",
                "admission_kernel_ready_score",
            ]
        }
        for exp, v in imports.items()
    }
    work.update(
        duration_s=(time.monotonic_ns() - start) / 1e9,
        clock=dict(
            started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
        ),
    )
    return work


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze current tests while the unchanged supervisor owns actual child receipts."""
    with patch.object(runner, "q", sys.modules[__name__]):
        plan: Json = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess, _exit"))
    pytest = str(ROOT / ".venv/bin/pytest")
    common = [pytest, "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    for spec in plan["commands"]:
        if spec["name"] == "coverage_custody_tests":
            spec["name"] = "owned_unit_and_private_CLI"
            spec["deadline"] = 480
        if spec["name"] == "consumer_E2E018":
            spec.update(
                name="consumer_E2E015_019",
                argv=common
                + [
                    "tests/python/test_source_boundary_7852.py",
                    "tests/python/test_experiment_7942_v689_sentence_labels.py",
                    "tests/python/test_primary_publication_7928.py",
                ],
                deadline=360,
            )
    for name, argv in [
        (
            "coverage_custody_tests",
            common
            + [
                "tests/python/test_coverage_custody_8262.py::" + name
                for name in [
                    "test_durable_measured_report",
                    "test_invalid_reports",
                    "test_rehashed_primitive_rejected",
                    "test_durable_provenance_negatives",
                ]
            ],
        ),
        ("view_component", common + [protocol.TEST, "-k", "focal or capture"]),
        ("admission_component", common + [protocol.TEST, "-k", "typed or hard_exit or admission"]),
    ]:
        plan["commands"].insert(
            2, dict(name=name, argv=argv, expected=0, deadline=480, scope="owned")
        )
    return plan


def reduce(work: Json, receipts: list[Json], coverage_ok: bool, cold_ok: bool) -> Json:
    """Require each component's primitives and current checks, never a copied success flag."""
    value = prior.reduce(work, receipts, coverage_ok, cold_ok)
    passed = {r["name"] for r in receipts if r["passed"] and r["exit_code"] == 0}
    scoped = {
        "current_contract_tests",
        "coverage_custody_tests",
        "view_component",
        "admission_component",
    }
    core = (
        bool(receipts)
        and all(r["passed"] for r in receipts if r["name"] not in scoped)
        and coverage_ok
        and cold_ok
    )
    failed = work["failures"]

    def ready(name: str) -> bool:
        """A missing operand can invalidate only the component that needs it."""
        return core and not any(g["component"] in {name, "shared"} for g in failed)

    evidence = work["protocol_work"]["evidence"]
    contract = int(
        ready("contract") and "current_contract_tests" in passed and work["contract"]["activated"]
    )
    coverage = int(
        ready("coverage")
        and "coverage_custody_tests" in passed
        and work["component_imports"]["8262"]["coverage_custody_ready_score"] == 1
    )
    protocol_ok = not any(g["component"] == "protocol" for g in failed)
    view = int(
        ready("view")
        and protocol_ok
        and "view_component" in passed
        and bool(work["protocol_work"]["tokenizer"])
        and evidence.get("capture", {}).get("completed_count") == 8
    )
    admission = int(
        ready("admission")
        and protocol_ok
        and "admission_component" in passed
        and evidence.get("positive_control_passed", False)
        and not work["protocol_work"]["owned_failure"]
    )
    owned = value["required_checks_passed"] and not work["protocol_work"]["owned_failure"]
    verdict = "disqualified" if not owned else "blocked" if failed else "circular_positive"
    for i, row in enumerate(value["rows"]):
        row["unit_id"] = f"exp{8276 + i}"
    for key in ["natural_shape_control_rows", "learnable_control_rows", "decision_control_rows"]:
        value["rows"].extend(
            dict(r, evidence_scope="current_private_oracle_control") for r in evidence.get(key, [])
        )
    captured = evidence.get("capture", {})
    for row, slot in zip(captured.get("rows", []), captured.get("intended_slots", [])):
        expected = "completed" if slot["mode"] == "valid" else "escalated"
        correct = row["status"] == expected
        value["rows"].append(
            dict(
                row,
                capture_status=row["status"],
                status="completed",
                completed=True,
                arm="scripted_capture",
                condition=slot["mode"],
                numerator=int(correct),
                denominator=1,
                failed=not correct,
                censored=False,
                excluded=False,
                independent_source_count=0,
            )
        )
    for row in work["history"]:
        missing = row["disposition"] == "cascade_skip_absent_primary"
        value["rows"].append(
            dict(
                row,
                unit_id="history-" + row["task_id"],
                arm="historical_disposition",
                condition=row["disposition"],
                status="censored" if missing else "completed",
                completed=not missing,
                failed=False,
                censored=missing,
                excluded=False,
                numerator=None if missing else 1,
                denominator=1,
                independent_source_count=0,
            )
        )
    value.update(
        intended_count=len(value["rows"]),
        completed_count=sum(r["status"] == "completed" for r in value["rows"]),
        failed_count=sum(bool(r.get("failed")) for r in value["rows"]),
        censored_count=sum(bool(r.get("censored")) for r in value["rows"]),
        excluded_count=sum(bool(r.get("excluded")) for r in value["rows"]),
    )
    value.update(
        experiment_id=8276,
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (failed[0]["artifact_field"] if verdict == "blocked" else "current_contract_readiness"),
        verdict_class=verdict,
        required_checks_passed=owned,
        verifier_is_oracle=True,
        current_contract_ready_score=contract,
        coverage_custody_ready_score=coverage,
        view_kernel_ready_score=view,
        admission_kernel_ready_score=admission,
        authority_snapshots=work["contract"]["authority_snapshots"],
        frozen_science_sha256=PIN,
        source_role_manifest=work["source_role_manifest"],
        frozen_head_config=work["frozen_head_config"],
        historical_canary=work["historical_canary"],
        component_validation_receipts={
            name: [r for r in receipts if r["name"] == command]
            for name, command in [
                ("contract", "current_contract_tests"),
                ("coverage", "coverage_custody_tests"),
                ("view", "view_component"),
                ("admission", "admission_component"),
            ]
        },
        tokenizer_receipt=work["protocol_work"]["tokenizer"],
        component_primitive_reference=dict(
            path=str(
                Path(work["protocol_work"]["code_config_hashes"][0]["path"]).parents[1]
                / "primitive_evidence.json"
            )
        )
        if evidence
        else {},
        acceptance_gates=dict(
            owned_validation=owned,
            current_contract=bool(contract),
            coverage_custody=bool(coverage),
            view_kernel=bool(view),
            admission_kernel=bool(admission),
            scientific_benefit=False,
        ),
    )
    return dict(value)


def build(
    work: Json,
    raw: Path,
    receipts: list[Json],
    binding: Json,
    cold: list[Json],
    scratch: Path,
    health: Json,
) -> Json:
    """Retain current protocol primitives without importing their fixture readiness."""
    oracle = protocol.build(work["protocol_work"], raw / "protocol", receipts, fixture=True)
    oracle_path = raw / "protocol" / "oracle_mechanics.json"
    atomic_json(oracle_path, oracle)
    value: Json = BASE_BUILD(work, raw, receipts, binding, cold, scratch, health)
    value["component_primitive_reference"] = dict(
        path=str(oracle_path), sha256=sha256_file(oracle_path)
    )
    value["field_principles"]["component_primitive_reference"] = (
        "Current private oracle requests and causal controls are replayed independently; their readiness headlines are not imported."
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild from frozen authority and original components after private cleanup."""
    value = json.loads(path.read_bytes())
    checksum = value.pop("reproducibility_checksum")
    if checksum != canonical_hash(value):
        raise ValueError("candidate_checksum")
    if (value["experiment_id"], value["task_id"], value["milestone"], value["run_date"]) != (
        8276,
        TASK,
        MILESTONE,
        "20261008",
    ):
        raise ValueError("invocation_identity")
    for ref in [
        value["work_reference"],
        value["component_primitive_reference"],
        *value["source_artifact_hashes"],
        *value["code_config_hashes"],
        *value["raw_shard_hashes"],
    ]:
        require_reference(ref)
    for receipt in value["validation_receipts"] + value["cold_replay_rows"]:
        for prefix in ["stdout", "stderr"]:
            if sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]:
                raise ValueError("validation_stream_hash")
    work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    measured = (
        custody.replay(value["coverage_command_receipt"])
        if value["coverage_command_receipt"]
        else {}
    )
    if (
        measured != value["owned_statement_counts"]
        or not value["scratch_removed"]
        or Path(value["scratch_path"]).exists()
    ):
        raise ValueError("durable_coverage_reduction")
    rebuilt = reduce(
        work,
        value["validation_receipts"],
        bool(measured),
        bool(value["cold_replay_rows"]) and all(r["passed"] for r in value["cold_replay_rows"]),
    )
    rebuilt.pop("component_primitive_reference")
    if any(value[k] != v for k, v in rebuilt.items()):
        raise ValueError("primitive_readiness_drift")
    refs = work["refs"]
    with TemporaryDirectory(prefix="carnot8276-replay-") as directory:
        if refs[0]["exists"]:
            paths = [
                Path(r.get("snapshot_path", Path(directory) / f"absent-{i}"))
                for i, r in enumerate(refs[:3])
            ]
            actual = assess(paths, Path(directory))
            for key in ["activated", "planning_matched", "canonical_tasks_sha256", "contract_rows"]:
                if actual[key] != work["contract"][key]:
                    raise ValueError("authority_reduction_drift")
            if parse_design(paths[0].read_text(), milestone=MILESTONE)[1] != work["tasks"]:
                raise ValueError("full_task_primitive_drift")
    for row in work["history"]:
        ref = next(r for r in refs if r["path"] == row["path"])
        source = json.loads(Path(ref["snapshot_path"]).read_bytes()) if ref["exists"] else {}
        if row.get("honest_verdict") != source.get("honest_verdict") or row["source_counts"] != {
            k: source.get(k) for k in row["source_counts"]
        }:
            raise ValueError("historical_primitive_drift")
    if not protocol.replay(Path(value["component_primitive_reference"]["path"])):
        raise ValueError("protocol_primitive_drift")
    if value["MODEL_SPECS"] or value["model_invocation_counts"] != protocol.ZERO_INVOCATION_COUNTS:
        raise ValueError("current_model_provenance")
    return dict(passed=True, owned_statement_counts=measured)


def main(argv: list[str] | None = None) -> int:
    """Expose the current thin CLI while reusing qualified children and publication."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--scripted-peer" in args or "--typed-roster" in args:
        return int(protocol.main(args))
    with (
        patch.object(runner, "q", sys.modules[__name__]),
        patch.object(runner, "manifest", manifest),
        patch.object(runner, "replay", replay),
        patch.object(runner, "build", build),
    ):
        return int(runner.main(args))
