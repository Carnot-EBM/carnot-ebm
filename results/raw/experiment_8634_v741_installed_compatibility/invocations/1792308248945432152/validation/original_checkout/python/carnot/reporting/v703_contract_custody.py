"""REQ-REPORT-8123: bind real scheduling bytes without claiming scientific benefit.

The same authority reader checks planning and activation. Historical failures
remain separate from reusable inputs so administrative failure cannot erase data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import v700_contract_custody as base
from carnot.reporting import v701_contract_custody as schema
from carnot.reporting import v702_contract_custody as previous
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v698_fixture_consumer_contract import failure

Json = dict[str, Any]
ROOT = previous.ROOT
Binder = previous.Binder
authority = previous.authority
NAME = "experiment_8123_v703_contract_custody"
TASK = "exp8123-contract-custody"
MILESTONE = "2026.10.703"
DESIGN = previous.DESIGN
MODULE = "python/carnot/reporting/v703_contract_custody.py"
RUNNER = previous.RUNNER
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_contract_custody_8123.py"
COVERAGE_PATHS = [MODULE, CLI]
CHECK_PATHS = COVERAGE_PATHS
PRESERVED = "openspec/change-proposals/research-roadmap-v702-preserved-20261004.md"
ALTERNATE = "results/experiment_8113_radial_decision_fit.json"
SKIPS = {8113, 8114, 8115, 8117}
QUALIFIED = {
    8111: ["methods_ready_score", "stream_input_ready_score"],
    8118: ["acquisition_cost_ready_score"],
    8121: ["hardware_boundary_ready_score"],
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts make waiting visible without implying unobserved inference."""
    print(f"[exp8123] phase={phase} completed={completed} pending={pending}", flush=True)


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Use the strict shipped parser so malformed prose cannot gain authority."""
    return base.assess(
        design, staged, active, raw, milestone=MILESTONE, first_id=8123, count=13, task=TASK
    )


def historical(root: Path, binder: Any, *, fixture: bool) -> Json:
    """Reconstruct actual task outcomes; a failed custody receipt supplies only bytes.

    Qualification reads each input's own validator and primitives. Large failed
    primaries remain frozen evidence, never material in current inference counters.
    """
    result: Json = dict(
        historical_dispositions=[],
        qualified_inputs={},
        authorities={},
        kernel_ready=False,
        controls_ready=False,
        public_kernel={},
        historical_controls=[],
    )
    admin = root / "results/experiment_8110_v702_contract_custody.json"
    if fixture:
        admin = root / "results/experiment_8110_fixture.json"
    try:
        receipt = binder.read(admin)
        snaps = receipt["authority_snapshots"]
        for role in ["active", "design"]:
            result["authorities"][role] = binder.bind(
                Path(snaps[role]["snapshot_path"]), snaps[role]["sha256"]
            )
        result["authorities"]["preserved_design"] = binder.bind(root / PRESERVED)
        activated = yaml.safe_load(
            Path(result["authorities"]["active"]["snapshot_path"]).read_bytes()
        )
        tasks = activated["tasks"]
        binder.require(admin, "historical_milestone", "2026.10.702", activated["milestone"])
        binder.require(
            admin,
            "historical_tasks_digest",
            receipt["canonical_tasks_sha256"],
            authority.tasks_digest(tasks),
        )
        binder.require(
            admin,
            "historical_sequence",
            list(range(8110, 8123)),
            [int(t["id"].split("-")[0][3:]) for t in tasks],
        )
        log_path = root / "ops/conductor-log.md"
        log = Path(binder.bind(log_path)["snapshot_path"]).read_text()
    except (OSError, ValueError, KeyError, TypeError) as error:
        binder.failures.append(
            failure(admin, "historical_authority_readable", True, str(error), TASK)
        )
        return result
    for index, task in enumerate(tasks):
        n = 8110 + index
        progress("historical_before", index, 13 - index)
        path = root / (ALTERNATE if n == 8113 else task["deliverable"])
        skipped = n in SKIPS
        lines = [s for s in log.splitlines() if task["title"][:48] in s]
        row: Json = dict(
            task_id=task["id"],
            path=str(path),
            primary_present=not skipped,
            conductor_skipped=skipped,
            conductor_log_rows=lines,
            no_retry_unchanged_outcome=True,
            sha256=None,
            verdict_class="blocked",
            honest_verdict="complete_blocked_conductor_skip",
        )
        try:
            binder.require(
                log_path,
                f"conductor_disposition_{n}",
                True,
                bool(lines) and any(("GATE_BLOCK" if skipped else "| OK |") in s for s in lines),
            )
            if not skipped or n == 8113:
                value = binder.read(path)
                if value.get("schema") == "blocked_gate_check_v1" and skipped:
                    binder.require(path, "experiment", n, value.get("experiment"))
                    binder.require(path, "title", task["title"], value.get("title"))
                else:
                    binder.require(path, "task_id", task["id"], value.get("task_id"))
                ref = next(r for r in binder.refs if r["path"] == str(path))
                row.update(
                    sha256=ref["sha256"],
                    verdict_class=value.get("verdict_class", "blocked"),
                    honest_verdict="complete_blocked_conductor_skip"
                    if skipped
                    else value["honest_verdict"],
                    producer_honest_verdict=value["honest_verdict"],
                    required_checks_passed=value.get("required_checks_passed"),
                    gate_check_summary=value.get("gate_check_summary", []),
                    historical_MODEL_SPECS=value.get("MODEL_SPECS", []),
                    historical_model_invocation_counts=value.get("model_invocation_counts", {}),
                )
                if skipped and value.get("failed_field"):
                    row["gate_check_summary"] = [
                        dict(
                            check=value["failed_field"],
                            artifact_field=value["failed_field"],
                            upstream=value["failed_upstream"],
                            path=value["failed_evidence_path"],
                            hash=value["failed_evidence_sha256"],
                            op=value["failed_operator"],
                            expected=value["failed_expected"],
                            observed=value["failed_observed"],
                            passed=False,
                        )
                    ]
                if n in QUALIFIED:
                    report = binder.terminal_evidence(path, value)
                    for field, expected in [
                        ("required_checks_passed", True),
                        ("flagged_adversarial", False),
                        *[(f, 1) for f in QUALIFIED[n]],
                    ]:
                        binder.require(path, field, expected, value.get(field))
                    binder.require(path, "terminal.report.passed", True, report.get("passed"))
                    for shard in value["raw_shard_hashes"]:
                        binder.bind(
                            Path(shard.get("snapshot_path", shard["path"])), shard["sha256"]
                        )
                    result["qualified_inputs"][str(n)] = dict(
                        path=str(path),
                        sha256=ref["sha256"],
                        ready_fields=QUALIFIED[n],
                        scope="historical_exposed_development_only",
                    )
        except (OSError, ValueError, KeyError, TypeError) as error:
            binder.failures.append(
                failure(path, "historical_operand_readable", True, str(error), task["id"])
            )
        result["historical_dispositions"].append(row)
        progress("historical_after", index + 1, 12 - index)
    return result


def measure(
    root: Path, design: Path, staged: Path, active: Path, raw: Path, *, fixture: bool = False
) -> Json:
    """Record exact prerequisites and phase spans, keeping current model activity empty."""
    start = time.monotonic_ns()
    binder = Binder(raw / "inputs", task=TASK)
    progress("preconditions_before")
    named = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/v685_authority_lifecycle.py",
        "ops/exclusion_manifest.yaml",
        "scripts/experiments/experiment_8110_v702_contract_custody.py",
        "tests/python/test_contract_custody_8110.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        MODULE,
        CLI,
        TEST,
        RUNNER,
    ]
    paths = [] if fixture else [root / p for p in named]
    paths += [ROOT / ".venv/bin" / p for p in ["python", "pytest", "coverage", "ruff", "mypy"]]
    for path in paths:
        try:
            binder.bind(path)
        except previous.InputFailure:
            pass
    boundary = time.monotonic_ns()
    progress("preconditions_after", len(paths), 0)
    contract = assess(design, staged, active, raw / "authority")
    history_start = time.monotonic_ns()
    progress("authority_after", len(contract["contract_rows"]), 13 - len(contract["contract_rows"]))
    history = historical(root, binder, fixture=fixture)
    atomic_json(
        raw / "primitive_rows.json",
        dict(
            authority_rows=contract["contract_rows"],
            historical_dispositions=history["historical_dispositions"],
        ),
    )
    end = time.monotonic_ns()
    boundaries = [start, boundary, history_start, end]
    return dict(
        contract=contract,
        history=history,
        refs=binder.refs,
        preconditions_checked=binder.observations + contract["gate_check_summary"],
        failures=binder.failures + contract["gate_check_summary"],
        started_ns=start,
        ended_ns=end,
        owner_pid=os.getpid(),
        fixture=fixture,
        root=str(root),
        phase_spans=[
            dict(
                phase=name,
                start_s=(boundaries[i] - start) / 1e9,
                end_s=(boundaries[i + 1] - start) / 1e9,
            )
            for i, name in enumerate(["preconditions", "authority", "historical_custody"])
        ],
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Reduce administrative checks only; scientific negatives remain historical."""
    receipts = [
        dict(r, normal_exit=r["exit_code"] >= 0 and not r.get("timed_out", False))
        if "exit_code" in r
        else r
        for r in receipts
    ]
    value = schema.build(work, raw, receipts)
    owned = value["required_checks_passed"]
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if value["gate_check_summary"]
        else "circular_positive"
        if work["fixture"]
        else "null"
    )
    for key in [
        "promised_but_unscheduled_ids",
        "public_kernel",
        "historical_controls",
        "radial_kernel_ready_score",
        "historical_controls_ready_score",
    ]:
        value.pop(key)
    for i, row in enumerate(value["rows"]):
        row.update(
            unit_id=f"exp{8123 + i}",
            source="V703_authority",
            source_id="V703_authority",
            source_cluster_id="V703_authority",
        )
    value["gate_check_summary"] = [
        dict(r, artifact_field=r.get("artifact_field", r.get("field")))
        for r in value["gate_check_summary"]
    ]
    value.update(
        experiment_id=8123,
        experiment=8123,
        task_id=TASK,
        milestone=MILESTONE,
        title="V703 contract custody",
        verdict_class=verdict,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            value["gate_check_summary"][0]["check"]
            if value["gate_check_summary"]
            else "contract_custody"
        ),
        verifier_is_oracle=int(work["fixture"]),
        independent_generalization_score=0,
        random_seed=70323,
        call_ledger=[],
        qualified_historical_inputs=work["history"]["qualified_inputs"],
        historical_authority_snapshots=work["history"]["authorities"],
        cited_upstream_artifacts=[
            dict(r, scope="historical_provenance_only")
            for r in work["history"]["historical_dispositions"]
        ],
        sample_size_budget=dict(
            administrative_tasks=13,
            historical_primaries=9,
            historical_gate_skips=4,
            independent_scientific_units=0,
        ),
        methodology_note="Compare complete V703 task bytes with strict authority parsing; independently preserve V702 primaries and conductor skips. No science, training, model loading or service benchmark is measured.",
    )
    value["raw_shard_hashes"].append(
        dict(path=str(raw / "primitive_rows.json"), sha256=sha256_file(raw / "primitive_rows.json"))
    )
    value["field_principles"] = {
        k: f"Record {k} so administrative custody cannot imply scientific benefit." for k in value
    }
    value["field_principles"].update(
        contract_ready_score="Complete authority agreement and normal owned validation qualify scheduling only.",
        cited_upstream_artifacts="Historical Qwen calls never enter current invocation counts.",
    )
    return value


def replay(path: Path) -> bool:
    """Independently rebuild frozen inputs so rehashing a forged aggregate cannot pass."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "work.json").read_text())
        refs = (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + [r for r in value["authority_snapshots"].values() if r["exists"]]
        )
        for ref in refs:
            if sha256_file(Path(ref.get("snapshot_path", ref.get("path")))) != ref["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        for label, ref in work["code_snapshots"].items():
            if sha256_file(Path(ref["snapshot_path"])) != value["code_config_hashes"][label]:
                return False
        with TemporaryDirectory(prefix="carnot-8123-replay-") as directory:
            private = Path(directory)
            snaps = value["authority_snapshots"]
            paths = [
                Path(snaps[k].get("snapshot_path", private / k))
                for k in ["design", "staged", "active"]
            ]
            contract = assess(*paths, private / "authority")
            for field in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]:
                if contract[field] != work["contract"][field]:
                    return False
            binder = previous.FrozenBinder(private / "inputs", work["refs"])
            if historical(Path(work["root"]), binder, fixture=work["fixture"]) != work["history"]:
                return False
        primitives = json.loads((raw / "primitive_rows.json").read_text())
        return (
            primitives
            == dict(
                authority_rows=work["contract"]["contract_rows"],
                historical_dispositions=work["history"]["historical_dispositions"],
            )
            and build(work, raw, value["validation_receipts"]) == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
