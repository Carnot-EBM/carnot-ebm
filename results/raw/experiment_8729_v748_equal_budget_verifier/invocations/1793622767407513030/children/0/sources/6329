"""Isolate training publication routes without changing math (REQ-REPORT-7941)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from types import ModuleType
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.publication_qualification_7928 import terminal_checks
from carnot.verify import energy_fit_7930 as custody
from carnot.verify import training_qualification_7904 as prior
from carnot.verify import training_qualification_7916 as qualified

ROOT, RUNTIME = custody.ROOT, custody.RUNTIME
CLI = "scripts/experiments/experiment_7941_v689_training_publication.py"
OWNED = (
    "python/carnot/verify/training_publication_7941.py",
    "python/carnot/verify/training_publication_7941_run.py",
    CLI,
)
INCLUDES = ",".join("*/" + name for name in OWNED)
START = time.monotonic()


def progress(phase: str, units: int = 0) -> None:
    """Show real elapsed work so child supervision can distinguish progress from silence."""
    print(
        f"[exp7941] phase={phase} completed={units} elapsed_s={time.monotonic() - START:.3f}",
        flush=True,
    )


def library() -> ModuleType:
    """Give checkpoints current code custody while retaining qualified numerical callables."""
    value = qualified.runtime()
    value.DEPENDENCIES = tuple(
        dict.fromkeys(
            (
                *value.DEPENDENCIES,
                *custody.OWNED,
                *OWNED,
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/publication_qualification_7928.py",
                "scripts/conductor_gates.py",
                "scripts/in_process_doc_reconcile.py",
            )
        )
    )
    value.progress = progress
    return value


def authenticate(runtime: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """External malformed receipts block the run rather than create an owned fit failure."""
    try:
        return custody.authenticate(runtime, custody.UPSTREAM)
    except custody.custody.InputBlocked:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row = custody.custody.operand(runtime, "runtime.valid_structure", "==", True, str(exc))
        row["upstream_id"] = "exp7916-training-qualification"
        raise custody.custody.InputBlocked([row]) from exc


def base(sources: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Each producer owns its identity; inherited fixture claims cannot become natural benefit."""
    value = prior.base(7941, "20260930", sources)
    value.update(
        {
            "task_id": "exp7941-training-publication",
            "milestone": "2026.09.689",
            "execution_date": "20260930",
            "historical_fixture_date": "20260929",
            "honest_verdict": "complete_blocked_runtime"
            if failures
            else "complete_circular_positive_training_publication",
            "verdict_class": "blocked" if failures else "circular_positive",
            "gate_check_summary": failures,
            "runtime_ready_score": 0,
            "training_publication_ready_score": 0,
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "inference_substrate_class": "no_model_load",
            "execution_venue": "host",
            "claim_scope": "private_fixture_oracle_agreement; natural_cohorts=exposed_development",
            "training_entrypoint": {
                "path": CLI,
                "callable": "carnot.verify.energy_fit_7930.fit_heads",
                "fixture_callable": "carnot.verify.natural_training.fit",
                "argv": [
                    str(ROOT / ".venv/bin/python"),
                    CLI,
                    "--date",
                    "20260930",
                    "--fixture",
                    "--output",
                    "<private-primary>",
                ],
                "downstream_producer": 7943,
            },
            "isolated_route_rows": [],
            "fixture_checkpoint_manifest": None,
            "consumer_selection_rows": [],
            "primary_resolution_receipt": {},
            "terminal_validation_sidecar_path": None,
        }
    )
    value["acceptance_gate_results"]["calibration"] = None
    value["resolved_imports"].update(
        {
            "carnot.verify." + Path(name).stem: str(ROOT / name)
            for name in (*OWNED[:2], *custody.OWNED[:2])
        }
    )
    value["field_principles"].update(
        {
            "training_publication_ready_score": "Every isolated required route, owned check, cold replay and nonempty full new-source coverage must pass.",
            "runtime_ready_score": "Current numerical dependency hashes and Exp7916's exact primary attestation qualify; owned failures close readiness.",
            "isolated_route_rows": "Actual child exits and expected failure reasons distinguish blocked publication from fitted success.",
            "fixture_checkpoint_manifest": "Fresh fixture checkpoint bytes belong to this producer; Exp7930 heads are never promoted.",
            "consumer_selection_rows": "Both unchanged consumers must select each private primary's actual path and hash.",
            "training_entrypoint": "Freeze the executable fixture route and full fitting callable for downstream Exp7943 custody.",
            "gate_check_summary": "External blocks retain upstream identity, exact path/hash and each failed operand.",
            "acceptance_gate_results": "Validity and readiness concern mechanics; probability, calibration, decision benefit, retention and efficiency remain unmeasured.",
        }
    )
    return value


def seal(value: dict[str, Any], output: Path, fault: str = "") -> dict[str, Any]:
    """Check final bytes, then test both actual readers after nested sidecars get newer mtimes."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["primary_resolution_receipt"] = {
        "path": str(raw / "primary_resolution.json"),
        "binding": "external exact final byte hash",
    }
    value["duration_s"] = time.monotonic() - START
    for key in value:
        value["field_principles"].setdefault(
            key,
            "Bind producer, exact bytes, primitive units and exposed role; no natural benefit is inferred.",
        )
    staging = raw / "candidate.json"
    atomic_json(staging, value)
    report = terminal_checks(staging, raw)
    if not report["passed"] or report["flagged_adversarial"]:
        value.update(
            honest_verdict="complete_disqualified_terminal_validation",
            verdict_class="disqualified",
            flagged_adversarial=report["flagged_adversarial"],
            training_publication_ready_score=0,
            runtime_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=0)
    if fault == "conflict":
        atomic_json(output.with_name("experiment_7941_conflict.json"), value)
    receipt = publish_primary(
        output, value, lambda p: {"passed": False} if fault == "reject" else terminal_checks(p, raw)
    )
    atomic_json(
        raw / "terminal_validation.json",
        {
            "candidate_sha256": receipt["primary_sha256"],
            "validator_sidecar_path": receipt["sidecar_path"],
        },
    )
    os.utime(receipt["sidecar_path"], ns=(output.stat().st_mtime_ns + 10**9,) * 2)
    selected = reader_receipt(
        value["task_id"],
        output.parent,
        field="training_publication_ready_score",
        expected=value["training_publication_ready_score"],
    )
    if not selected["passed"] or selected["gate_sha256"] != receipt["primary_sha256"]:
        raise ValueError("primary consumer mismatch")
    atomic_json(raw / "primary_resolution.json", selected)
    progress("published", len(value["rows"]))
    return receipt


def produce(runtime: Path, output: Path, fault: str = "") -> int:
    """A missing runtime publishes a blocked artifact; a valid one runs a real bounded fixture fit."""
    progress("authenticate")
    raw = output.parent / "raw" / output.stem
    try:
        sources, dependencies, qualification = authenticate(runtime)
    except custody.custody.InputBlocked as exc:
        value = base([], exc.operands)
        progress("source evidence blocked")
    else:
        value = base(sources, [])
        q = library()
        upstream = q.make_fixture(raw / "fixture_input")
        progress("before_fixture_fit")
        fitted = q.fit_score(upstream, raw / "attempt.json", raw / "fit", 7941, "20260930")
        progress("after_fixture_fit", len(fitted["rows"]))
        value.update(
            {
                key: fitted[key]
                for key in (
                    "rows",
                    "fixture_prediction_rows",
                    "prediction_rows_path",
                    "prediction_rows_sha256",
                    "checkpoint_manifest_path",
                    "checkpoint_manifest_sha256",
                    "checkpoint_identity",
                    "trained_head_specs",
                    "sample_size_budget",
                    "phase_spans",
                )
            }
        )
        value["source_artifact_hashes"] += fitted["source_artifact_hashes"]
        value.update(
            runtime_ready_score=1,
            training_dependency_hashes={**dependencies, **q.dependency_hashes()},
            fixture_checkpoint_manifest=value["checkpoint_manifest_path"],
        )
        value["preconditions_checked"] = {
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "training_runtime_ready_score": 1,
        }
        value["historical_required_failures"] = qualification["historical_required_failures"]
    seal(value, output, fault)
    return 0


def replay(path: Path, recheck: bool = False) -> None:
    """Reconstruct predictions from primitive bytes and reject any stale primary binding."""
    value = json.loads(path.read_text())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    read_bound_sidecar(path, Path(terminal["validator_sidecar_path"]))
    if value["verdict_class"] == "blocked":
        if (
            not value["gate_check_summary"]
            or value["runtime_ready_score"]
            or value["training_publication_ready_score"]
        ):
            raise ValueError("blocked gate drift")
    elif value.get("fixture_prediction_rows"):
        library().replay(path)
    else:
        rows = value["isolated_route_rows"]
        if not rows or not all(row["passed"] for row in rows):
            raise ValueError("route reduction failed")
        from carnot.verify import training_publication_7941_run as driver

        raw = path.parent / "raw" / path.stem
        frozen = json.loads(Path(value["validation_command_manifest_path"]).read_text())
        receipts = value["validation_receipts"]
        if [row["name"] for row in receipts] != [row["name"] for row in frozen["commands"]]:
            raise ValueError("primitive receipt mismatch")
        for receipt in receipts:
            log = Path(receipt["log_path"])
            if not log.is_absolute():
                log = ROOT / log
            if sha256_file(log) != receipt["log_sha256"]:
                raise ValueError("primitive log hash drift")
        coverage = json.loads((raw / "coverage.json").read_text())["files"]
        counts = {name: item["summary"] for name, item in coverage.items()}
        if (
            counts != value["coverage_statement_counts"]
            or int(driver.reduce_readiness(receipts, counts))
            != value["training_publication_ready_score"]
        ):
            raise ValueError("primitive readiness drift")
        if (
            value["sample_size_budget"]["completed"] != len(rows)
            or value["sample_size_budget"]["independent"] != 0
        ):
            raise ValueError("primitive denominator drift")
        for row in value["consumer_selection_rows"]:
            if sha256_file(Path(row["gate_path"])) != row["gate_sha256"]:
                raise ValueError("route primary hash drift")
            replay(Path(row["gate_path"]))
    for source in value["source_artifact_hashes"]:
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("source artifact hash drift")
    if recheck:
        report = terminal_checks(path, path.parent / "raw" / path.stem)
        if not report["passed"] or report["flagged_adversarial"]:
            raise ValueError("terminal recheck failed")
    progress("cold_replay_passed", len(value["rows"]))


from carnot.verify.training_publication_7941_run import freeze  # noqa: E402
