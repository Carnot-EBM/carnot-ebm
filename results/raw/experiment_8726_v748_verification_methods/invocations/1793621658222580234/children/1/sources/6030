"""Bind qualified fitting to current custody (REQ-VERIFY-7943, REQ-REPORT-7943)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time
from types import ModuleType, SimpleNamespace
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.publication_qualification_7928 import terminal_checks
from carnot.verify import energy_fit_7930 as old
from carnot.verify import energy_fit_7930_run as old_run
from carnot.verify import training_publication_7941 as qualified

ROOT, RUNTIME, UPSTREAM = old.ROOT, old.RUNTIME, old.UPSTREAM
PUBLICATION = ROOT / "results/experiment_7941_v689_training_publication.json"
TASK = "exp7943-energy-fit"
CLI = "scripts/experiments/experiment_7943_v689_energy_fit.py"
OWNED = (
    "python/carnot/verify/energy_fit_7943.py",
    "python/carnot/verify/energy_fit_7943_validation.py",
    CLI,
)
INCLUDES = ",".join("*/" + name for name in OWNED)
START = time.monotonic()


def progress(phase: str, event: str = "boundary", units: int = 0) -> None:
    """Measured work counters let the supervisor distinguish computation from silence."""
    print(
        f"[exp7943] phase={phase} event={event} completed={units} elapsed_s={time.monotonic() - START:.3f}",
        flush=True,
    )


def isolated(path: Path) -> ModuleType:
    """A private namespace changes configuration without changing historical consumers."""
    spec = importlib.util.spec_from_file_location("carnot.verify._7943_" + path.stem, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def authenticate(
    publication: Path, runtime: Path, upstream: Path
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Primary operands and exact callable bytes must qualify before numerical work."""

    def require(field: str, expected: Any, observed: Any) -> None:
        if expected != observed:
            row = old.custody.operand(publication, field, "==", expected, observed)
            row["upstream_id"] = "exp7941-training-publication"
            raise old.custody.InputBlocked([row])

    try:
        require("artifact.exists", True, publication.is_file())
        value = json.loads(publication.read_text())
        for field, expected in (
            ("experiment_id", 7941),
            ("run_date", "20260930"),
            ("training_publication_ready_score", 1),
            ("runtime_ready_score", 1),
            ("flagged_adversarial", False),
        ):
            require(field, expected, value.get(field))
        require(
            "terminal_eligible",
            True,
            value.get("verdict_class") in ("positive", "circular_positive", "null")
            and str(value.get("honest_verdict", "")).startswith("complete_"),
        )
        require(
            "training_entrypoint.callable",
            "carnot.verify.energy_fit_7930.fit_heads",
            value["training_entrypoint"]["callable"],
        )
        dependencies = value["training_dependency_hashes"]
        require("dependencies.nonempty", True, bool(dependencies))
        for name, expected in dependencies.items():
            path = ROOT / name
            require("dependency." + name, expected, sha256_file(path) if path.is_file() else None)
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
        require("terminal.candidate_sha256", sha256_file(publication), terminal["candidate_sha256"])
        report = read_bound_sidecar(publication, Path(terminal["validator_sidecar_path"]))["report"]
        require("terminal.passed", True, report["passed"] and not report["flagged_adversarial"])
    except old.custody.InputBlocked:
        raise
    except (ValueError, OSError, KeyError, TypeError) as exc:
        row = old.custody.operand(publication, "publication.valid_structure", "==", True, str(exc))
        row["upstream_id"] = "exp7941-training-publication"
        raise old.custody.InputBlocked([row]) from exc
    references, runtime_dependencies, runtime_value = qualified.authenticate(runtime)
    source = next(
        (
            r
            for r in runtime_value["source_artifact_hashes"]
            if Path(r["path"]).resolve() == upstream.resolve()
        ),
        {},
    )
    require(
        "qualified_source.sha256",
        source.get("sha256"),
        sha256_file(upstream) if upstream.is_file() else None,
    )
    return (
        [
            {
                "role": "qualified_publication_primary",
                "path": str(publication),
                "sha256": sha256_file(publication),
            },
            *references,
        ],
        {**runtime_dependencies, **dependencies},
        value,
    )


def current_fit(*args: Any) -> dict[str, Any]:
    """The same trained parameters acquire current producer custody before saving."""
    head: dict[str, Any] = old.natural_training.fit(*args)
    head.update(experiment_id=7943, task_id=TASK, run_date="20260930", fresh_current_fit=True)
    return head


def seal(value: dict[str, Any], output: Path) -> None:
    """Only inspected final bytes reach readers; sidecars bind that final hash."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = {
        "path": str(raw / "primary_resolution.json"),
        "binding": "external final byte hash",
    }
    value["duration_s"] = time.monotonic() - START
    value["acceptance_gate_results"].setdefault("calibration", None)
    for key in value:
        value["field_principles"].setdefault(
            key,
            "Bind executing producer, exact bytes, measured units and exposed roles; running is not a benefit claim.",
        )
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    report = terminal_checks(candidate, raw)
    if not report["passed"] or report["flagged_adversarial"]:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            energy_fit_ready_score=0,
            flagged_adversarial=report["flagged_adversarial"],
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
        atomic_json(candidate, value)
        report = terminal_checks(candidate, raw)
        value["flagged_adversarial"] = report["flagged_adversarial"]
        atomic_json(candidate, value)
    receipt = publish_primary(output, value, lambda path: terminal_checks(path, raw))
    atomic_json(
        raw / "terminal_validation.json",
        {
            "candidate_sha256": receipt["primary_sha256"],
            "validator_sidecar_path": receipt["sidecar_path"],
        },
    )
    os.utime(receipt["sidecar_path"], ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    selected = reader_receipt(
        TASK,
        output.parent,
        field="energy_fit_ready_score",
        expected=value["energy_fit_ready_score"],
    )
    if not selected["passed"] or selected["gate_sha256"] != receipt["primary_sha256"]:
        raise ValueError("primary consumer mismatch")
    atomic_json(raw / "primary_resolution.json", selected)
    progress("published", receipt["primary_sha256"], len(value["rows"]))


def driver(publication: Path) -> ModuleType:
    """Reuse fitting orchestration in isolated namespaces with current task configuration."""
    from carnot.verify import energy_fit_7943_validation as validation

    engine = isolated(Path(old.__file__))
    run = isolated(Path(old_run.__file__))
    engine.START, engine.OWNED, engine.progress = START, OWNED, progress
    engine.natural_training = SimpleNamespace(**{**vars(old.natural_training), "fit": current_fit})
    run.core, run.INCLUDES = engine, INCLUDES
    run._original_freeze = run.freeze
    run.TESTS = ("tests/python/test_energy_fit_7943.py",)
    run.CONSUMERS = (
        *old_run.TESTS,
        *old_run.CONSUMERS,
        "tests/python/test_training_publication_7941.py",
    )
    original_library, original_base = engine.library, run.base
    original_execute, original_enrich = run.execute, run.enrich
    cached: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    def library(upstream: Path, output: Path) -> ModuleType:
        q = original_library(upstream, output)
        q.START, q.OWNED, q.progress = START, list(OWNED), progress
        original_score = q.score

        def score(*args: Any) -> Any:
            path, controls = original_score(*args)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row.update(experiment_id=7943, task_id=TASK, run_date="20260930")
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            return path, controls

        q.score = score
        return q

    def base(
        q: Any, upstream: Path, sources: list[dict[str, Any]], failures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        value: dict[str, Any] = original_base(q, upstream, sources, failures)
        historical = json.loads((ROOT / "results/experiment_7930_v688_energy_fit.json").read_text())
        value.update(
            experiment_id=7943,
            task_id=TASK,
            milestone="2026.09.689",
            execution_date="20260930",
            historical_fixture_date="20260929",
        )
        value["historical_required_failures"] += historical["historical_required_failures"]
        value["historical_required_failures"] += [
            {"experiment_id": 7930, **r}
            for r in historical["validation_receipts"]
            if not r["passed"] and r["scope"] != "repository_health"
        ]
        value["repository_health"]["historical_exp7930_verdict"] = historical["honest_verdict"]
        value["resolved_imports"].update(
            {"carnot.verify." + Path(name).stem: str(ROOT / name) for name in OWNED[:2]}
        )
        value["acceptance_gate_results"]["calibration"] = None
        value["validation_receipts"] = list(cached)
        value["observed_child_commands"] = [row.get("argv", []) for row in cached]
        return value

    def freeze(private: Path, raw: Path) -> Any:
        path, commands = validation.freeze(run, private, raw)
        state.update(commands=commands, raw=raw)
        return path, commands

    def authenticate_current(runtime: Path, upstream: Path) -> Any:
        sources, dependencies, value = authenticate(publication, runtime, upstream)
        progress("private_qualification", "before success, blocked, failure, replay and recheck")
        cached.extend(
            original_execute(
                [c for c in state["commands"] if c.name in validation.PREFIT], state["raw"]
            )
        )
        if not all(row["passed"] for row in cached):
            raise ValueError("isolated publication qualification failed")
        progress("private_qualification", "after passed routes", len(cached))
        return sources, dependencies, value

    def execute(commands: list[Any], raw: Path) -> list[dict[str, Any]]:
        retained = cached if any(c.name in validation.PREFIT for c in commands) else []
        return [
            *retained,
            *original_execute([c for c in commands if c.name not in validation.PREFIT], raw),
        ]

    def enrich(*args: Any) -> dict[str, Any]:
        value: dict[str, Any] = original_enrich(*args)
        validation.diagnostics(value)
        value["preconditions_checked"].update(
            training_publication_ready_score=1,
            runtime_ready_score=1,
            private_publication_qualified_before_fit=True,
        )
        for spec in value["trained_head_specs"]:
            spec.update(experiment_id=7943, task_id=TASK, run_date="20260930")
        return value

    engine.library, engine.authenticate = library, authenticate_current
    run.base, run.freeze, run.execute, run.enrich, run.publish = base, freeze, execute, enrich, seal
    return run


def fixture_library() -> ModuleType:
    """A short current fit exercises the same numerical and checkpoint library before natural fitting."""
    q = qualified.library()
    q.DEPENDENCIES = (*q.DEPENDENCIES, *OWNED)
    q.natural_training = SimpleNamespace(**{**vars(old.natural_training), "fit": current_fit})
    original = q.checkpoint_identity

    def identity(*args: Any) -> dict[str, Any]:
        value: dict[str, Any] = original(*args)
        value.update(experiment_id=7943, task_id=TASK, run_date="20260930")
        return value

    q.checkpoint_identity, q.progress = (
        identity,
        lambda phase, units=0: progress(phase, "fixture", units),
    )
    return q


def fixture(publication: Path, runtime: Path, output: Path) -> int:
    """Keep fixture mechanics separate from natural 27-head measurement readiness."""
    raw = output.parent / "raw" / output.stem
    q = old.library(UPSTREAM, output)
    run = driver(publication)
    try:
        sources, _, _ = authenticate(publication, runtime, UPSTREAM)
    except old.custody.InputBlocked as exc:
        value = run.base(q, UPSTREAM, [], exc.operands)
        progress("preconditions", "source evidence blocked")
    else:
        lib = fixture_library()
        upstream = lib.make_fixture(raw / "fixture_input")
        progress("fixture_fit", "before")
        fitted = lib.fit_score(upstream, raw / "attempt.json", raw / "fit", 7943, "20260930")
        value = run.base(q, UPSTREAM, sources, [])
        for key in (
            "rows",
            "fixture_prediction_rows",
            "source_artifact_hashes",
            "checkpoint_manifest_path",
            "checkpoint_manifest_sha256",
            "checkpoint_identity",
            "prediction_rows_path",
            "prediction_rows_sha256",
            "trained_head_specs",
            "sample_size_budget",
            "phase_spans",
        ):
            value[key] = fitted[key]
        value["source_artifact_hashes"] += sources
        for row in value["rows"]:
            row.update(experiment_id=7943, task_id=TASK, run_date="20260930")
        value.update(
            honest_verdict="complete_circular_positive_publication_fixture",
            verdict_class="circular_positive",
            verifier_is_oracle=True,
            claim_scope="private_fixture_oracle_agreement",
            fixture_ready_score=1,
        )
        value["acceptance_gate_results"]["validity"] = True
        Path(value["prediction_rows_path"]).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in value["rows"])
        )
        value["prediction_rows_sha256"] = sha256_file(Path(value["prediction_rows_path"]))
        progress("fixture_fit", "after", len(value["rows"]))
    seal(value, output)
    return 0


def replay(path: Path, recheck: bool = False) -> None:
    """Reconstruct primitive predictions and reject drift in final primary custody."""
    value = json.loads(path.read_text())
    if path.name.startswith("experiment_7943_"):
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
        read_bound_sidecar(path, Path(terminal["validator_sidecar_path"]))
    if value.get("fixture_prediction_rows"):
        fixture_library().replay(path)
    else:
        engine = driver(PUBLICATION).core
        engine.replay(path)
    if recheck and not terminal_checks(path, path.parent / "raw" / path.stem)["passed"]:
        raise ValueError("terminal recheck failed")
    progress("replay", "passed", len(value["rows"]))
