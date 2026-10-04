"""REQ-REPORT-8110: preserve schedule and evidence without granting science credit.

Each qualified historical input has its own gate. A failed fitted head cannot
invalidate public stream features or turn numerical fixtures into learning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import v701_contract_custody as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v698_fixture_consumer_contract import failure

Json = dict[str, Any]
ROOT = previous.ROOT
Binder = previous.Binder
InputFailure = previous.InputFailure
authority = previous.authority
NAME = "experiment_8110_v702_contract_custody"
TASK = "exp8110-contract-custody"
MILESTONE = "2026.10.702"
DESIGN = previous.DESIGN
MODULE = "python/carnot/reporting/v702_contract_custody.py"
RUNNER = previous.RUNNER
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_contract_custody_8110.py"
COVERAGE_PATHS = [MODULE, CLI]
CHECK_PATHS = COVERAGE_PATHS
CAPSTONE = "results/experiment_8109_v701_capstone.json"
KERNEL = "results/experiment_8085_v700_radial_memory_kernel.json"
CAPSTONE_HASH = "sha256:feb1a3b1c56b456f30c80f07aa83bd623b58266c9823540039e8bf9f4b779775"
STAGE_COMMIT = "34917eb7864d9a7e081251d7d058c03e668d3255"
STAGE_HASH = "sha256:4abb9bcb9fafb38647e09cf3991499b5bc6e001a27d9707ce4e18a73bd569fd0"
QUALIFIED = {
    8098: ("roles", "cohort_ready_score"),
    8102: ("stream", "stream_capture_ready_score"),
    8085: ("kernel", "kernel_ready_score"),
    8105: ("native", "native_kernel_ready_score"),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so custody cannot imply an unobserved model call."""
    print(f"[exp8110] phase={phase} completed={completed} pending={pending}", flush=True)


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Use the shipped parameterized reader, including consumed staging paths."""
    result = previous.previous.assess(
        design, staged, active, raw, milestone=MILESTONE, first_id=8110, count=13, task=TASK
    )
    if not result["tasks"] and active.is_file():
        observed = yaml.safe_load(active.read_bytes())
        if isinstance(observed, dict) and isinstance(observed.get("tasks"), list):
            result["tasks"] = observed["tasks"]
            result["canonical_tasks_sha256"] = authority.tasks_digest(observed["tasks"])
    if design.is_file() and "## Exact task contract" not in design.read_text():
        result["gate_check_summary"] = [
            failure(design, "design_exact_task_contract", True, False, TASK)
        ]
    return result


def saved_stage(root: Path, raw: Path) -> Path:
    """Recover observed committed bytes, rather than invent a missing staging file."""
    raw.mkdir(parents=True, exist_ok=True)
    argv = ["git", "show", STAGE_COMMIT + ":research-roadmap-next.yaml"]
    progress("before_saved_stage_subprocess")
    started = time.monotonic()
    done = subprocess.run(argv, cwd=root, capture_output=True, timeout=30)
    log = raw / "committed_stage.log"
    log.write_bytes(done.stdout + done.stderr)
    atomic_json(
        raw / "committed_stage_receipt.json",
        dict(
            argv=argv,
            exit_code=done.returncode,
            normal_exit=done.returncode >= 0,
            duration_s=time.monotonic() - started,
            log_path=str(log),
            log_sha256=sha256_file(log),
        ),
    )
    progress("after_saved_stage_subprocess")
    if done.returncode != 0 or sha256_file(log) != STAGE_HASH:
        raise InputFailure("committed_staging_bytes")
    return log


def terminal(path: Path, value: Json, binder: Binder, *, quarantined: bool = False) -> Json:
    """Preserve stale quarantine bindings; require exact binding for qualified data."""
    side = binder.read(Path(value["terminal_validation_sidecar_path"]))
    publication = side.get("publication", side)
    report_path = Path(publication["sidecar_path"])
    report = binder.read(report_path)
    digest = next(r["sha256"] for r in binder.refs if r["path"] == str(path))
    bound = publication.get("primary_sha256") == report.get("primary_sha256") == digest
    if not quarantined:
        binder.require(path, "terminal.primary_sha256", digest, publication.get("primary_sha256"))
        binder.require(
            report_path, "terminal.report.primary_sha256", digest, report.get("primary_sha256")
        )
        binder.require(
            report_path,
            "terminal.sidecar_location",
            True,
            report_path.is_relative_to(path.parent / "raw" / path.stem),
        )
    binder.logs(side)
    binder.logs(report)
    return dict(report=report["report"], bound=bound)


def qualify(path: Path, value: Json, evidence: Json, binder: Binder, *, fixture: bool) -> Json:
    """Authenticate only named reusable inputs; negative science stays historical."""
    n = value["experiment_id"]
    branch, field = QUALIFIED[n]
    for key, expected in [
        (field, 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        binder.require(path, key, expected, value.get(key))
    binder.require(path, "terminal.report.passed", True, evidence["report"].get("passed"))
    refs = list(value.get("raw_shard_hashes", []))
    if n == 8098:
        roles = value["role_manifests"]
        binder.require(
            path,
            "public_roles",
            ["evaluation", "fit", "retention", "stream", "tune"],
            sorted(roles),
        )
        refs.extend(roles.values())
    if n == 8102:
        refs.extend([value["stream_feature_manifest"], value["retention_feature_manifest"]])
    if n == 8105:
        loaded = value["loaded_binding_receipt"]
        binder.require(path, "loaded_binding.actual_loaded", True, loaded.get("actual_loaded"))
        binder.require(
            path, "loaded_binding.sha256", value["native_library_sha256"], loaded.get("sha256")
        )
        refs.append(dict(path=value["native_library_path"], sha256=value["native_library_sha256"]))
    for ref in refs:
        binder.bind(Path(ref.get("snapshot_path", ref["path"])), ref["sha256"])
    if not fixture:
        for label, digest in value.get("code_config_hashes", {}).items():
            if label.startswith(("python/", "crates/", "scripts/experiments/", "tests/python/")):
                binder.bind(ROOT / label, digest)
    return dict(
        branch=branch,
        ready=True,
        primary_path=str(path),
        qualified_refs=refs,
        historical_model_invocation_counts=value.get("model_invocation_counts", {}),
        historical_MODEL_SPECS=value.get("MODEL_SPECS", []),
        verdict_class=value["verdict_class"],
    )


def historical(root: Path, binder: Binder, *, fixture: bool) -> Json:
    """Retain every scheduled disposition, including skipped and quarantined work."""
    result: Json = dict(
        historical_dispositions=[],
        qualified_inputs={},
        kernel_ready=False,
        controls_ready=False,
        public_kernel={},
        historical_controls=[],
    )
    for branch, _ in QUALIFIED.values():
        result[branch + "_ready"] = False
    cap_path = root / CAPSTONE
    try:
        cap = binder.read(cap_path, None if fixture else CAPSTONE_HASH)
        terminal(cap_path, cap, binder)
        tasks = cap["task_contract"]
        binder.require(
            cap_path,
            "historical_task_sequence",
            list(range(8097, 8110)),
            [int(t["id"].split("-")[0][3:]) for t in tasks],
        )
        old = {r["task_id"]: r for r in cap["task_dispositions"]}
        log_path = root / "ops/conductor-log.md"
        log = Path(binder.bind(log_path)["snapshot_path"]).read_text()
    except (OSError, ValueError, KeyError, TypeError) as error:
        binder.failures.append(
            failure(cap_path, "historical_manifest_readable", True, str(error), TASK)
        )
        return result
    for index, task in enumerate(tasks):
        progress("historical_before", index, 13 - index)
        n = 8097 + index
        original = old[task["id"]]
        path = cap_path if n == 8109 else Path(original["path"])
        row = dict(original, task_id=task["id"], no_retry_unchanged_outcome=True)
        try:
            if n in [8100, 8101, 8103, 8104]:
                lines = original["conductor_log_rows"]
                binder.require(
                    log_path,
                    f"conductor_skip_{n}",
                    True,
                    bool(lines) and all(s in log for s in lines),
                )
                if original.get("sha256"):
                    binder.bind(path, original["sha256"])
                row.update(primary_present=False, conductor_skipped=True)
            else:
                value = cap if n == 8109 else binder.read(path, original["sha256"])
                evidence = terminal(path, value, binder, quarantined=n == 8099)
                row.update(
                    path=str(path),
                    sha256=next(r["sha256"] for r in binder.refs if r["path"] == str(path)),
                    primary_present=True,
                    conductor_skipped=False,
                    verdict_class=value["verdict_class"],
                    honest_verdict=value["honest_verdict"],
                    historical_hash_binding_passed=evidence["bound"],
                    historical_validation_passed=evidence["report"].get("passed"),
                    terminal_validation_sidecar_path=value["terminal_validation_sidecar_path"],
                )
                if n in QUALIFIED:
                    qualified = qualify(path, value, evidence, binder, fixture=fixture)
                    branch = qualified["branch"]
                    result[branch + "_ready"] = True
                    result["qualified_inputs"][branch] = qualified
        except (OSError, ValueError, KeyError, TypeError) as error:
            binder.failures.append(
                failure(path, "historical_evidence_readable", True, str(error), task["id"])
            )
        result["historical_dispositions"].append(row)
        progress("historical_after", index + 1, 12 - index)
    path = root / KERNEL
    try:
        value = binder.read(path, None if fixture else previous.HISTORY_HASHES[8085])
        evidence = terminal(path, value, binder)
        result["qualified_inputs"]["kernel"] = qualify(
            path, value, evidence, binder, fixture=fixture
        )
        result["kernel_ready"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        binder.failures.append(failure(path, "kernel_evidence_readable", True, str(error), TASK))
    return result


def measure(
    root: Path, design: Path, staged: Path, active: Path, raw: Path, *, fixture: bool = False
) -> Json:
    """Freeze prerequisites and real phase boundaries with no current inference."""
    started = time.monotonic_ns()
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
        "ops/exclusion_manifest.yaml",
        DESIGN,
        "python/carnot/reporting/v685_authority_lifecycle.py",
        "python/carnot/reporting/roadmap_contract.py",
        "scripts/experiments/experiment_8097_v701_contract_custody.py",
        "openspec/change-proposals/research-roadmap-v701-preserved-20261004.md",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_primary_publication_7928.py",
        MODULE,
        RUNNER,
        CLI,
        TEST,
    ]
    paths = [] if fixture else [root / n for n in named]
    paths += [ROOT / ".venv/bin" / n for n in ["python", "pytest", "coverage", "ruff", "mypy"]]
    for path in paths:
        try:
            binder.bind(path)
        except InputFailure:
            pass
    before_authority = time.monotonic_ns()
    progress("preconditions_after", len(paths), 0)
    contract = assess(design, staged, active, raw / "authority")
    if not fixture and not staged.is_file() and root == ROOT:
        try:
            saved = saved_stage(root, raw)
            staged_value = yaml.safe_load(saved.read_bytes())
            active_value = yaml.safe_load(active.read_bytes())
            contract["authority_snapshots"]["saved_staged"] = authority._snapshot(
                saved, saved.read_bytes(), raw / "authority", "saved_staged"
            )
            binder.require(
                saved,
                "saved_staging_tasks_sha256",
                authority.tasks_digest(active_value["tasks"]),
                authority.tasks_digest(staged_value["tasks"]),
            )
            binder.require(saved, "saved_staging_milestone", MILESTONE, staged_value["milestone"])
            contract["saved_staging_validated"] = True
            contract["saved_staging_commit"] = STAGE_COMMIT
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            binder.failures.append(
                failure(staged, "committed_staging_bytes", STAGE_HASH, str(error), TASK)
            )
    before_history = time.monotonic_ns()
    progress("authority_after", len(contract["contract_rows"]), 13 - len(contract["contract_rows"]))
    history = historical(root, binder, fixture=fixture)
    atomic_json(
        raw / "primitive_rows.json",
        dict(
            authority_rows=contract["contract_rows"],
            historical_dispositions=history["historical_dispositions"],
        ),
    )
    ended = time.monotonic_ns()
    boundaries = [started, before_authority, before_history, ended]
    spans = [
        dict(
            phase=name,
            start_s=(boundaries[i] - started) / 1e9,
            end_s=(boundaries[i + 1] - started) / 1e9,
        )
        for i, name in enumerate(["preconditions", "authority", "historical_custody"])
    ]
    return dict(
        contract=contract,
        history=history,
        refs=binder.refs,
        preconditions_checked=binder.observations + contract["gate_check_summary"],
        failures=binder.failures + contract["gate_check_summary"],
        started_ns=started,
        ended_ns=ended,
        owner_pid=os.getpid(),
        fixture=fixture,
        root=str(root),
        phase_spans=spans,
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Reuse the qualified schema, then reduce this milestone's thirteen tasks."""
    value = previous.build(work, raw, receipts)
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
    for field in [
        "promised_but_unscheduled_ids",
        "public_kernel",
        "historical_controls",
        "radial_kernel_ready_score",
        "historical_controls_ready_score",
    ]:
        value.pop(field)
    rows = value["rows"]
    for index, row in enumerate(rows):
        row.update(
            unit_id=f"exp{8110 + index}",
            source_id="V702_authority",
            source="V702_authority",
            source_cluster_id="V702_authority",
        )
    value.update(
        experiment_id=8110,
        experiment=8110,
        task_id=TASK,
        milestone=MILESTONE,
        title="V702 contract custody",
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
        random_seed=70210,
        call_ledger=[],
        scope_reduction_compliance=dict(
            qualified_data_excludes_8099=True,
            historical_tasks_preserved=13,
            science_credit=0,
            no_retry_unchanged_outcomes=True,
        ),
        qualified_historical_inputs=work["history"]["qualified_inputs"],
        sample_size_budget=dict(
            administrative_tasks=13,
            historical_primaries=9,
            historical_gate_skips=4,
            independent_scientific_units=0,
        ),
        methodology_note="Authenticate V702 full executable authority and V701 terminal dispositions; independently qualify historical roles, stream, numerical kernel and native binding. No model or learning benefit is measured.",
    )
    for branch, _ in QUALIFIED.values():
        value["historical_" + branch + "_ready_score"] = int(
            owned and work["history"][branch + "_ready"]
        )
    value["gate_check_summary"] = [
        dict(g, artifact_field=g.get("artifact_field", g.get("field")))
        for g in value["gate_check_summary"]
    ]
    value["saved_staging_validated"] = work["contract"].get("saved_staging_validated", False)
    value["saved_staging_commit"] = work["contract"].get("saved_staging_commit")
    value["raw_shard_hashes"].extend(
        dict(path=str(raw / name), sha256=sha256_file(raw / name))
        for name in ["primitive_rows.json", "committed_stage_receipt.json"]
        if (raw / name).is_file()
    )
    value["acceptance_gates"] = dict(
        contract="Full prompts, visible order and immutable activation prevent invented tasks.",
        historical_roles="Exact public role bytes prevent role drift and evaluator leakage.",
        historical_stream="Authenticated cached public features prevent stale Qwen calls becoming current activity.",
        historical_kernel="Clean numerical fixtures qualify arithmetic only, with zero natural-data benefit.",
        historical_native="Exact loaded binding bytes and current source prevent an unbuilt native claim.",
        publication="Passing owned checks and normal validator exits prevent unchecked reader exposure.",
    )
    value["field_principles"] = {
        k: f"Record {k} so administrative custody cannot imply scientific benefit." for k in value
    }
    value["field_principles"].update(
        verdict_class="External unchanged blocks are terminal; owned failures disqualify readiness.",
        historical_dispositions="Nine primaries and four skips preserve all thirteen scheduled tasks.",
        canonical_tasks_sha256="Hash full executable prompts and failure history, not just titles.",
        call_ledger="Only current calls belong here; historical Qwen receipts are cited separately.",
        source_cluster_id="Thirteen administrative units share one authority and create no independent sources.",
    )
    return value


class FrozenBinder(Binder):
    """Read authenticated saved operands so replay survives live-path deletion."""

    def __init__(self, raw: Path, refs: list[Json]):
        super().__init__(raw, task=TASK)
        self.saved = {r["path"]: r for r in refs}

    def bind(self, path: Path, digest: str | None = None) -> Json:
        ref = self.saved.get(str(path))
        self.require(path, "resource_exists", True, ref is not None)
        assert ref is not None
        self.require(
            path, "sha256", digest or ref["sha256"], sha256_file(Path(ref["snapshot_path"]))
        )
        if ref not in self.refs:
            self.refs.append(ref)
        return ref


def replay(path: Path) -> bool:
    """Rebuild historical qualifications and authority checks before comparing totals."""
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
        with TemporaryDirectory(prefix="carnot-8110-replay-") as directory:
            private = Path(directory)
            snapshots = value["authority_snapshots"]
            paths = [
                Path(snapshots[k].get("snapshot_path", private / k))
                for k in ["design", "staged", "active"]
            ]
            rebuilt = assess(*paths, private / "authority")
            if any(
                rebuilt[k] != work["contract"][k]
                for k in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]
            ):
                return False
            if "saved_staged" in snapshots:
                saved = Path(snapshots["saved_staged"]["snapshot_path"])
                staged_value = yaml.safe_load(saved.read_bytes())
                if (
                    staged_value["milestone"] != MILESTONE
                    or authority.tasks_digest(staged_value["tasks"])
                    != value["canonical_tasks_sha256"]
                ):
                    return False
            binder = FrozenBinder(private / "inputs", work["refs"])
            history = historical(Path(work["root"]), binder, fixture=work["fixture"])
            if history != work["history"]:
                return False
        return build(work, raw, value["validation_receipts"]) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False
