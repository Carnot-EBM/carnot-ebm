"""Reduce producer observations before granting any scientific credit.

REQ-REPORT-7914-V686. A complete audit can retain missing science and old failures.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.v685_capstone import reduce_rows
from carnot.reporting.v686_contract_methods import assess

ROOT = Path(__file__).resolve().parents[3]
OWNED = (
    "python/carnot/reporting/v686_capstone.py",
    "scripts/experiments/experiment_7914_v686_capstone.py",
)
TEST = "tests/python/test_experiment_7914_v686_capstone.py"
QUALIFIED = {"positive", "circular_positive", "null"}
COUNTS = (
    "intended",
    "eligible",
    "started",
    "completed",
    "failed",
    "censored",
    "excluded",
    "independent",
)
SCIENCE = {7906, 7907, 7908, 7909, 7910, 7912}


def progress(started: float, phase: str, units: int) -> None:
    """Flushed boundaries let an operator distinguish running work from a stalled child."""
    print(
        f"[exp7914] phase={phase} elapsed_s={time.monotonic() - started:.3f} completed_units={units}",
        flush=True,
    )


def read(path: Path) -> tuple[dict[str, Any], str | None]:
    """One byte read binds the parsed value to its exact evidence hash."""
    if not path.is_file():
        return {}, None
    raw = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return {}, digest
    return (value if isinstance(value, dict) else {}), digest


def operand(
    path: Path,
    digest: str | None,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    op: str = "==",
) -> dict[str, Any]:
    """Keep absent fields distinct from values that fail a declared gate."""
    return dict(
        upstream_id=upstream,
        artifact_path=str(path),
        artifact_hash=digest,
        artifact_field=field,
        op=op,
        expected=expected,
        observed=observed,
    )


def reduce_primitives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use family reducers, then audit causal order and issued prediction sets."""
    reduced = reduce_rows(rows)
    actions: dict[str, list[bool]] = defaultdict(list)
    delayed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    writes = 0
    for row in rows:
        if set(row.get("feature_fields", [])) & {"label", "human_label", "original_human_label"}:
            raise ValueError("label_feature")
        if "dependency_hashes" in row and row["dependency_hashes"] != row.get(
            "current_dependency_hashes"
        ):
            raise ValueError("stale_dependency")
        if "bank_write_step" in row:
            if (
                not row["prediction_step"]
                < row["feedback_step"]
                <= row["bank_write_step"]
                < row["later_decision_step"]
            ):
                raise ValueError("feedback_order")
            writes += 1
        if "issued_alpha" in row and row["issued_alpha"] != row.get(
            "feedback_issued_alpha", row["issued_alpha"]
        ):
            raise ValueError("issued_alpha_changed")
        if "tau" in row and "release_step" in row:
            if row["release_step"] < row["prediction_step"] + row["tau"]:
                raise ValueError("early_feedback")
        if "action" in row:
            actions[str(row.get("arm", "default"))].append(row["action"] == "abstain")
        if "prediction_set" in row:
            delayed[str(row["tau"])].append(row)
    reduced["abstention_by_arm"] = {
        arm: sum(values) / len(values) for arm, values in actions.items()
    }
    reduced["causal_writes"] = writes
    reduced["delayed_sets"] = {
        tau: {
            "issued": len(values),
            "released": sum("release_step" in r for r in values),
            "pending": sum("release_step" not in r for r in values),
            "coverage": sum(
                r["label"] in r["prediction_set"] for r in values if "release_step" in r
            )
            / max(1, sum("release_step" in r for r in values)),
            "mean_set_size": sum(len(r["prediction_set"]) for r in values) / len(values),
            "windows": len({str(r.get("window")) for r in values}),
        }
        for tau, values in delayed.items()
    }
    return reduced


def build_candidate(
    root: Path, design: Path, active: Path, date: str, publication: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Bind all twelve dispositions without treating conductor acceptance as benefit."""
    started = time.monotonic()
    start_ns = time.monotonic_ns()
    progress(started, "start", 0)
    if date != "20260930":
        raise ValueError("v686_date_changed")
    private = Path(tempfile.mkdtemp(prefix="carnot-7914-authority-"))
    authority = assess(design, root / "research-roadmap-next.yaml", active, private)
    tasks = (
        yaml.safe_load(active.read_bytes())["tasks"]
        if active.is_file()
        else yaml.safe_load(
            gzip.decompress((ROOT / "tests/fixtures/v686/active.yaml.gz").read_bytes())
        )["tasks"]
    )
    failures = list(authority["gate_check_summary"])
    sources = [
        {
            "path": row["source_path"],
            "sha256": row["sha256"],
            "role": role,
            "exposure": "administrative",
        }
        for role, row in authority["authority_snapshots"].items()
    ]
    observed: dict[str, tuple[dict[str, Any], str | None, Path]] = {}
    for task in tasks[:11]:
        path = root / task["deliverable"]
        data, digest = read(path)
        observed[task["id"]] = data, digest, path
        sources.append(
            dict(
                path=str(path),
                sha256=digest,
                role="declared_producer",
                exposure="exposed_development",
            )
        )
    progress(started, "resolved_paths_hashes_roles_operands", 11)
    outcomes, reductions, history = [], [], []
    for index, task in enumerate(tasks[:12]):
        number = 7903 + index
        data, digest, path = observed.get(task["id"], ({}, None, root / task["deliverable"]))
        state = (
            "self_administrative" if number == 7914 else str(data.get("verdict_class", "missing"))
        )
        receipt_path = (
            root
            / "results"
            / f"experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
        )
        receipt, receipt_hash = read(receipt_path) if not data and number != 7914 else ({}, None)
        if receipt.get("blocked_at_layer") == "conductor_pre_gate":
            state = "skipped"
            sources.append(
                dict(
                    path=str(receipt_path),
                    sha256=receipt_hash,
                    role="conductor_gate_receipt",
                    exposure="administrative",
                )
            )
        own = []
        if number != 7914 and not data:
            own.append(operand(path, digest, task["id"], "producer_exists", True, False))
        if data:
            for field, expected in (
                ("experiment_id", number),
                ("task_id", task["id"]),
                ("milestone", "2026.09.686"),
                ("run_date", date),
                ("MODEL_SPECS", task["MODEL_SPECS"]),
                ("flagged_adversarial", False),
            ):
                if data.get(field, "missing_field") != expected:
                    own.append(
                        operand(
                            path,
                            digest,
                            task["id"],
                            field,
                            expected,
                            data.get(field, "missing_field"),
                        )
                    )
            history.extend(data.get("historical_required_failures", []))
            history.append(
                dict(
                    upstream_id=task["id"],
                    path=str(path),
                    sha256=digest,
                    honest_verdict=data.get("honest_verdict"),
                    required_failures=data.get("gate_check_summary", []),
                    resolved=False,
                )
            )
        for gate in task.get("gated_on", []):
            prior, prior_hash, prior_path = observed.get(
                gate["upstream"], ({}, None, root / "missing")
            )
            actual = prior.get(
                gate["artifact_field"], "missing_field" if prior else "missing_source"
            )
            passed = (
                actual == gate["value"]
                if gate["op"] == "=="
                else actual in gate["value"]
                if gate["op"] == "in"
                else False
            )
            if not passed:
                failures.append(
                    operand(
                        prior_path,
                        prior_hash,
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["value"],
                        actual,
                        gate["op"],
                    )
                )
        raw = data.get("rows", [])
        try:
            reduced = reduce_primitives(raw)
        except (ValueError, TypeError, KeyError) as error:
            reduced = reduce_primitives([])
            own.append(
                operand(path, digest, task["id"], "primitive_audit", "valid rows", str(error))
            )
        if own and state in QUALIFIED:
            state = "disqualified"
        failures.extend(own)
        unit = "board_obligation" if number == 7913 else "family_observation"
        reductions.append(
            dict(
                upstream_id=task["id"],
                path=str(path),
                sha256=digest,
                unit=unit,
                primitive_rows=raw,
                **reduced,
            )
        )
        outcomes.append(
            dict(
                upstream_id=task["id"],
                task_id=task["id"],
                path=str(path),
                hash=digest,
                status=state,
                eligible=state in QUALIFIED and not own,
                conductor_gate_receipt=dict(
                    path=str(receipt_path),
                    sha256=receipt_hash,
                    gates_evaluated=receipt.get("gates_evaluated", []),
                ),
                producer_budget=data.get("sample_size_budget"),
                exposure="exposed_development",
            )
        )
        progress(started, "independent_reduction", index + 1)
    previous_path = root / "results/experiment_7902_v685_capstone.json"
    previous, previous_hash = read(previous_path)
    history.append(
        dict(
            upstream_id="exp7902-capstone",
            path=str(previous_path),
            sha256=previous_hash,
            honest_verdict=previous.get("honest_verdict"),
            coverage="243/377",
            resolved=False,
            required_failures=previous.get("gate_check_summary", []),
        )
    )
    sources.append(
        dict(
            path=str(previous_path),
            sha256=previous_hash,
            role="historical_failure",
            exposure="historical",
        )
    )
    science_ready = authority["activated"] and all(
        r["eligible"] for r in outcomes if int(r["task_id"][3:7]) in SCIENCE
    )
    gaps = {
        "FR-12": (7904, 7905, 7906, 7907, 7908),
        "FR-11": (7909, 7910),
        "FR-05/FR-08/NFR-01": (7911, 7912, 7913),
    }
    decisions = {
        gap: dict(
            decision="blocked"
            if any(
                outcomes[n - 7903]["status"] in {"missing", "skipped", "blocked"} for n in members
            )
            else "disqualified"
            if any(not outcomes[n - 7903]["eligible"] for n in members)
            else "measured-null",
            required_producers=list(members),
            independent_benefit=False,
            continue_if="All declared producers pass owned checks and fresh primitive comparisons show benefit within preregistered bounds.",
            retire_if="The same verdict and failed operands recur without changed inputs or qualification.",
        )
        for gap, members in gaps.items()
    }
    counts = dict(zip(COUNTS, (12, 12, 12, 12, 0, 0, 0, 0), strict=True))
    publication = publication or {}
    gates = {
        key: bool(publication.get("gates", {}).get(key, {}).get("pass", False))
        for key in ("G1", "G2", "G3", "G4")
    }
    value = build_current_work_receipt(
        run_id="exp7914-20260930",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"work": "independent primitive reduction"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=[
            dict(
                phase="authority_and_reduction",
                start_s=0,
                end_s=time.monotonic() - started,
                completed_units=12,
            )
        ],
    )
    value.update(
        experiment_id=7914,
        task_id="exp7914-capstone",
        milestone="2026.09.686",
        run_date=date,
        honest_verdict="complete_null_independent_benefit_unshown"
        if science_ready
        else "complete_blocked_missing_science",
        verdict_class="null" if science_ready else "blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=outcomes,
        outcome_rows=outcomes,
        independent_reduction_rows=reductions,
        sample_size_budget={
            **counts,
            "unit": "task_disposition",
            "science_unit_budgets": {
                r["upstream_id"]: {
                    "unit": r["unit"],
                    "primitive_observations": r["intended"],
                    "families": r["independent_families"],
                    "scientifically_independent": 0,
                }
                for r in reductions
            },
        },
        acceptance_gate_results=dict(
            validity=True,
            readiness=1,
            probability_quality=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        random_seed=7914,
        source_artifact_hashes=sources,
        preconditions_checked=authority["gate_check_summary"],
        resolved_imports={
            "carnot.reporting.v686_capstone": str(Path(__file__).resolve()),
            "carnot.reporting.v685_capstone": str(Path(reduce_rows.__code__.co_filename).resolve()),
            "carnot.reporting.v686_contract_methods": str(
                Path(assess.__code__.co_filename).resolve()
            ),
        },
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        historical_required_failures=history,
        repository_health=dict(status="historical_backlog_open", affects_required_checks=False),
        verifier_is_oracle=False,
        claim_scope=dict(
            natural_data="exposed_development",
            fixture_agreement="circular_positive",
            gap_oracle_distinct="open",
            independent_benefit=False,
            DiffusionGemma="pending_actual_distinct_oracle_accuracy_and_efficiency_evidence",
        ),
        model_specs=[],
        target_model="none",
        model_invocation_counts=dict(loads=0, calls=0, tokens=0),
        trained_head_specs=[],
        capstone_execution_ready_score=1,
        gap_decisions=decisions,
        retirement_decisions=[
            dict(
                prior=item,
                identical_prior_failure=any(
                    r.get("honest_verdict") == item.get("verdict") for r in history
                ),
                decision="retire"
                if any(r.get("honest_verdict") == item.get("verdict") for r in history)
                else "retain_changed_scope",
            )
            for item in tasks[-1].get("prior_failures", [])
        ],
        authority_snapshots=authority["authority_snapshots"],
        canonical_tasks_sha256=authority["canonical_tasks_sha256"],
        activation_confirmed=authority["activated"],
        **gates,
        paper_ready=all(gates.values()),
        unmet_gates=[k for k, met in gates.items() if not met],
        publication_gate_results=publication,
        report_path="docs/research-notes/experiment_7914_v686_capstone.md",
    )
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=sources, rows=reductions, seed=7914)
    )
    value["field_principles"] = {
        key: "Bind actual producer bytes and their units; audit completion gives no independent scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        {
            f"acceptance_gate_results.{key}": "Own validation controls readiness; absent scientific measurement remains null."
            for key in value["acceptance_gate_results"]
        }
    )
    return value


def cold_replay(value: dict[str, Any], root: Path, design: Path, active: Path) -> list[str]:
    """A cold reader checks primitive reductions and byte custody without running science."""
    expected = build_candidate(
        root, design, active, "20260930", value.get("publication_gate_results")
    )
    errors = []
    for source in value["source_artifact_hashes"]:
        path = Path(source["path"])
        if (sha256_file(path) if path.is_file() else None) != source["sha256"]:
            errors.append("source_bytes_changed")
    for field in (
        "rows",
        "outcome_rows",
        "independent_reduction_rows",
        "sample_size_budget",
        "gap_decisions",
        "reproducibility_checksum",
    ):
        if value.get(field) != expected[field]:
            errors.append(f"{field}_changed")
    return sorted(set(errors))


def validation_commands(private: Path) -> list[CommandSpec]:
    """Freeze explicit private routes, identical coverage includes and bounded deadlines."""
    py, pytest, cov, ruff, mypy = (
        str(ROOT / ".venv/bin" / name) for name in ("python", "pytest", "coverage", "ruff", "mypy")
    )
    include = ",".join(str(ROOT / f) for f in OWNED)
    common = ("-q", "-n", "0", "-o", "addopts=", "--no-cov")
    fixture_root = private / "fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    for name, target in (("design.md", "design.md"), ("active.yaml", "research-roadmap.yaml")):
        (fixture_root / target).write_bytes(
            gzip.decompress((ROOT / f"tests/fixtures/v686/{name}.gz").read_bytes())
        )
    base = (
        "--date",
        "20260930",
        "--root",
        str(fixture_root),
        "--design",
        str(fixture_root / "design.md"),
        "--active",
        str(fixture_root / "research-roadmap.yaml"),
    )
    script = str(ROOT / OWNED[1])
    rows = [
        CommandSpec(
            "publication_gate", (py, "scripts/publication_gate.py", "--json"), "required", 60
        ),
        CommandSpec(
            "unit",
            (
                cov,
                "run",
                f"--data-file={private / 'unit.coverage'}",
                f"--include={include}",
                "-m",
                "pytest",
                TEST,
                *common,
                f"--basetemp={private / 'unit-temp'}",
            ),
            "required",
            180,
        ),
        CommandSpec(
            "consumers",
            (
                pytest,
                "tests/python/test_experiment_7902_v685_capstone.py",
                "tests/python/test_experiment_7890_v684_capstone.py",
                "tests/python/test_experiment_7877_v683_independent_audit.py",
                "tests/python/test_experiment_7878_v683_capstone.py",
                *common,
                f"--basetemp={private / 'consumers'}",
            ),
            "required",
            180,
        ),
    ]
    for name, args in (
        ("success", (*base, "--output", str(private / "success.json"), "--evidence-only")),
        (
            "block",
            (
                "--date",
                "20260930",
                "--root",
                str(private / "absent"),
                "--output",
                str(private / "block.json"),
                "--evidence-only",
            ),
        ),
        ("failure", ("--evidence-only",)),
        ("replay", (*base, "--cold-replay", str(private / "success.json"))),
    ):
        rows.append(
            CommandSpec(
                f"cli_{name}",
                (
                    cov,
                    "run",
                    f"--data-file={private / f'{name}.coverage'}",
                    f"--include={include}",
                    script,
                    *args,
                ),
                "expected_failure" if name == "failure" else "required",
                60,
            )
        )
    for name, test in (
        ("e2e015", "tests/python/test_source_boundary_7852.py"),
        ("e2e017", "tests/python/test_arc_supervisor_delta_7874.py"),
        ("e2e018_private", "tests/python/test_experiment_7891_v685_authority_lifecycle.py"),
    ):
        rows.append(
            CommandSpec(
                name, (pytest, test, *common, f"--basetemp={private / name}"), "required", 180
            )
        )
    for name, route in (("fixture", "--fixture-e2e"), ("replay", "--cold-replay")):
        rows.append(
            CommandSpec(
                f"e2e016_{name}",
                (
                    py,
                    "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                    "--date",
                    "20260930",
                    route,
                    str(private / "e2e016.json"),
                ),
                "required",
                180,
            )
        )
    rows.extend(
        [
            CommandSpec(
                "coverage_combine",
                (
                    cov,
                    "combine",
                    "--keep",
                    f"--data-file={private / 'combined.coverage'}",
                    *(
                        str(private / f"{name}.coverage")
                        for name in ("unit", "success", "block", "failure", "replay")
                    ),
                ),
                "required",
                60,
            ),
            CommandSpec(
                "coverage_report",
                (
                    cov,
                    "report",
                    f"--data-file={private / 'combined.coverage'}",
                    f"--include={include}",
                    "--show-missing",
                    "--fail-under=100",
                ),
                "required",
                60,
            ),
            CommandSpec("ruff", (ruff, "check", *OWNED, TEST), "required", 60),
            CommandSpec("format", (ruff, "format", "--check", *OWNED, TEST), "required", 60),
            CommandSpec("mypy", (mypy, "--strict", *OWNED), "required", 120),
            CommandSpec("spec", (py, "scripts/check_spec_coverage.py", TEST), "required", 60),
            CommandSpec(
                "repository_full_suite",
                (pytest, "tests/python", *common, f"--basetemp={private / 'full'}"),
                "diagnostic",
                180,
            ),
        ]
    )
    return rows


def qualify(root: Path, design: Path, active: Path, date: str, output: Path) -> int:
    """Seal finished children and check the exact bytes before atomic publication."""
    started = time.monotonic()
    private = Path(tempfile.mkdtemp(prefix="carnot-7914-validation-"))
    commands = validation_commands(private)
    dependency_paths = {
        *OWNED,
        TEST,
        "python/carnot/reporting/v685_capstone.py",
        "python/carnot/reporting/v685_authority_lifecycle.py",
        "python/carnot/reporting/v686_contract_methods.py",
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/experiment_7303_validation_scope.py",
        "scripts/publication_gate.py",
        "scripts/adversarial_verify.py",
        "scripts/verdict_row_consistency_lint.py",
        "scripts/check_spec_coverage.py",
        "tests/python/conftest.py",
        "pyproject.toml",
        "ops/exclusion_manifest.yaml",
    }
    manifest = dict(
        commands=[
            dict(
                name=c.name,
                argv=list(c.argv),
                expected_exit=2 if c.name == "cli_failure" else 0,
                failure_reason="--date" if c.name == "cli_failure" else None,
                deadline_s=c.timeout_s,
                scope=c.scope,
            )
            for c in commands
        ],
        measured_modules=list(OWNED),
        dependency_hashes={p: sha256_file(ROOT / p) for p in sorted(dependency_paths)},
    )
    checkpoint = canonical_hash(manifest)[7:]
    durable = root / "results/raw/experiment_7914_v686_capstone" / checkpoint
    durable.mkdir(parents=True, exist_ok=True)
    manifest_path = durable / "validation-command-manifest.json"
    atomic_json(manifest_path, manifest)
    value = build_candidate(root, design, active, date)
    receipts = run_commands(ROOT, commands, log_dir=private / "logs", heartbeat_s=30)
    for spec, row in zip(commands, receipts, strict=True):
        source = ROOT / row["log_path"]
        target = durable / (sha256_file(source)[7:] + ".log")
        shutil.copyfile(source, target)
        row.update(
            log_path=str(target),
            argv=list(spec.argv),
            expected_exit=2 if spec.name == "cli_failure" else 0,
            deadline_s=spec.timeout_s,
            classification=spec.scope,
        )
        row["passed"] = row["exit_code"] == row["expected_exit"] and (
            spec.name != "cli_failure" or "--date" in target.read_text()
        )
        if spec.name == "publication_gate" and row["passed"]:
            value["publication_gate_results"] = json.loads(target.read_text())
    for key in ("G1", "G2", "G3", "G4"):
        value[key] = bool(
            value["publication_gate_results"].get("gates", {}).get(key, {}).get("pass")
        )
    value.update(
        paper_ready=all(value[k] for k in ("G1", "G2", "G3", "G4")),
        unmet_gates=[k for k in ("G1", "G2", "G3", "G4") if not value[k]],
        validation_receipts=receipts,
        validation_command_manifest_path=str(manifest_path),
        observed_child_commands=[r["argv"] for r in receipts],
    )
    value["repository_health"]["current_full_suite"] = [
        r for r in receipts if r["classification"] == "diagnostic"
    ]
    required_failures = [
        r for r in receipts if r["classification"] != "diagnostic" and not r["passed"]
    ]
    value["gate_check_summary"].extend(
        operand(
            Path(r["log_path"]),
            r["log_sha256"],
            "exp7914-owned-validation",
            r["name"],
            r["expected_exit"],
            r["exit_code"],
        )
        for r in required_failures
    )
    if required_failures:
        value.update(
            honest_verdict="complete_disqualified_required_validation",
            verdict_class="disqualified",
            capstone_execution_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
    for snapshot in value["authority_snapshots"].values():
        if snapshot["exists"]:
            source = Path(snapshot["snapshot_path"])
            target = durable / source.name
            shutil.copyfile(source, target)
            snapshot["snapshot_path"] = str(target)
    sidecar = durable / "terminal-validation.json"
    value["terminal_validation_sidecar_path"] = str(sidecar)
    value["field_principles"]["terminal_validation_sidecar_path"] = (
        "Exact candidate hashes bind actual validator reports after children exit."
    )
    candidate = private / "candidate.json"
    value["ended_monotonic_ns"] = time.monotonic_ns()
    value["duration_s"] = (value["ended_monotonic_ns"] - value["started_monotonic_ns"]) / 1e9
    value["phase_spans"].append(
        dict(
            phase="owned_validation",
            start_s=value["phase_spans"][0]["end_s"],
            end_s=value["duration_s"],
            completed_units=len(receipts),
        )
    )
    for attempt in range(3):
        atomic_json(candidate, value)
        terminal = [
            CommandSpec(
                "adversarial_verify",
                (
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ),
                "terminal",
                60,
            ),
            CommandSpec(
                "strict_rows",
                (
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "terminal",
                60,
            ),
        ]
        checks = run_commands(
            ROOT, terminal, log_dir=private / f"terminal-{attempt}", heartbeat_s=30
        )
        report = json.loads((ROOT / checks[0]["log_path"]).read_text())
        flagged = bool(report.get("flagged_count", 0))
        terminal_failed = flagged or not all(r["passed"] for r in checks)
        for row in checks:
            source = ROOT / row["log_path"]
            target = durable / (sha256_file(source)[7:] + ".log")
            shutil.copyfile(source, target)
            row["log_path"] = str(target)
        atomic_json(
            sidecar,
            dict(
                candidate_sha256=sha256_file(candidate),
                validator_receipts=checks,
                adversarial_report=report,
            ),
        )
        if terminal_failed and attempt == 0:
            value.update(
                honest_verdict="complete_disqualified_terminal_verification",
                verdict_class="disqualified",
                capstone_execution_ready_score=0,
                flagged_adversarial=flagged,
            )
            value["acceptance_gate_results"].update(validity=False, readiness=0)
            value["gate_check_summary"].append(
                operand(
                    sidecar,
                    None,
                    "exp7914-owned-validation",
                    "terminal_validators",
                    "unflagged and passing",
                    report,
                )
            )
            continue
        if value["flagged_adversarial"] != flagged:
            value["flagged_adversarial"] = flagged
            continue
        break
    report_path = root / value["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# V686 capstone\n\nVerdict: "
        + value["honest_verdict"]
        + ".\n\n"
        + "\n".join(
            f"- {r['task_id']}: {r['status']} ({r['path']})." for r in value["outcome_rows"]
        )
        + "\n\n"
        + "\n".join(
            f"- {gap}: {decision['decision']}. {decision['continue_if']}"
            for gap, decision in value["gap_decisions"].items()
        )
        + "\n\nMissing science remains terminal blocked. Development gains cannot close GAP-ORACLE-DISTINCT. DiffusionGemma remains pending. Historical failures remain in the artifact.\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, json.loads(candidate.read_bytes()))
    progress(started, "published_checked_bytes", 12)
    return 0
