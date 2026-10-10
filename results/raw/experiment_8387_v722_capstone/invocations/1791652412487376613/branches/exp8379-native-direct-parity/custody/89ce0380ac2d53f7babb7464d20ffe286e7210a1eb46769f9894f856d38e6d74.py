"""REQ-REPORT-8379: bounded children produce checked terminal publication bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import native_direct_8379 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v718_replay_runner import audit

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8379-private"


def plan(private: Path) -> list[Json]:
    """Freeze commands and owned coverage before opening measured parity outputs."""
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "\n".join("    " + str(e.ROOT / p) for p in e.OWNED)
        + "\n"
    )
    os.environ["COVERAGE_RCFILE"] = str(config)
    os.environ["COVERAGE_FILE"] = str(private / ".coverage")
    python = str(e.ROOT / ".venv/bin/python")
    files = e.OWNED + [e.TEST]
    commands = [
        (
            "owned_tests",
            [
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "-q",
                "--cov",
                "--cov-config=" + str(config),
                e.TEST,
                "--basetemp=" + str(private / "unit"),
            ],
            600,
            "owned",
        ),
        (
            "private_E2E018_consumers",
            [
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_primary_publication_7928.py",
                "--basetemp=" + str(private / "e2e"),
            ],
            600,
            "owned",
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *files], 60, "owned"),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *files], 60, "owned"),
        ("strict_mypy", [str(e.ROOT / ".venv/bin/mypy"), "--strict", *e.OWNED], 120, "owned"),
        (
            "spec_coverage",
            [
                python,
                "scripts/check_spec_coverage.py",
                "--files",
                e.TEST,
                "crates/carnot-core/tests/direct_spline_8379.rs",
            ],
            60,
            "owned",
        ),
        (
            "coverage_combine",
            [python, "-m", "coverage", "combine", "--keep", "--append"],
            60,
            "owned",
        ),
        (
            "coverage_json",
            [python, "-m", "coverage", "json", "-o", str(private / "coverage.json")],
            60,
            "owned",
        ),
        ("coverage_100", [python, "-m", "coverage", "report", "--fail-under=100"], 60, "owned"),
        (
            "repository_health_once",
            [
                str(e.ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "global"),
            ],
            900,
            "global",
        ),
    ]
    frozen = [
        dict(name=name, argv=argv, deadline_s=deadline, scope=scope)
        for name, argv, deadline, scope in commands
    ]
    for path in json.loads(os.environ.get("CARNOT_8379_PRIOR_VALIDATION", "[]")):
        receipt = json.loads(Path(path).read_bytes())
        spec = dict(
            name=receipt["name"],
            argv=receipt["argv"],
            deadline_s=receipt["deadline_s"],
            scope=receipt["scope"],
            receipt_reference=e.reference(Path(path)),
        )
        frozen = [row for row in frozen if row["name"] != spec["name"]]
        frozen.append(spec)
    return frozen


def reuse(spec: Json) -> Json:
    """Reuse exact prior commands without rerunning the repository-wide suite."""
    ref = spec["receipt_reference"]
    if e.reference(Path(ref["path"])) != ref:
        raise ValueError("prior_receipt_hash")
    receipt: Json = json.loads(Path(ref["path"]).read_bytes())
    for stream in ("stdout", "stderr"):
        if e.reference(Path(receipt[stream + "_path"]))["sha256"] != receipt[stream + "_sha256"]:
            raise ValueError("prior_log_hash")
    return receipt


def validate(candidate: Path, raw: Path) -> Json:
    """Unchanged validators and typed finding consumers retain every observation."""
    cold = child(
        "cold",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
        raw,
        deadline=120,
    )
    found = audit(candidate, raw / "adversarial", {})
    rows = child(
        "strict_rows",
        [
            sys.executable,
            "-u",
            "scripts/verdict_row_consistency_lint.py",
            "--strict",
            str(candidate),
        ],
        raw,
        deadline=60,
    )
    value = json.loads(candidate.read_bytes())
    qualified = all(r["passed"] for r in (cold, found["receipt"], rows))
    quarantined = bool(
        value["verdict_class"] == "disqualified"
        and value["native_parity_ready_score"] == 0
        and value["required_checks_passed"] is False
        and value["flagged_adversarial"] is True
        and value["adversarial_findings"] == found["findings"]
    )
    return dict(
        passed=bool(cold["passed"] and rows["passed"] and (qualified or quarantined)),
        qualification_passed=qualified,
        checked_failure_publication=quarantined,
        checks=[cold, found["receipt"], rows],
        adversarial=found,
    )


def controls(candidate: Path, raw: Path) -> list[Json]:
    """A repaired top-level hash must not authorize invented native parity."""
    bad = deepcopy(json.loads(candidate.read_bytes()))
    bad["native_invocation_count"] += 1
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, bad)
    return [
        child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            raw / "controls",
            expected=expected,
            deadline=120,
        )
        for name, args, expected in [
            ("valid", ["--cold-replay", str(candidate)], 0),
            ("missing", ["--cold-replay", str(raw / "absent")], 1),
            ("error", ["--deliberate-error"], 1),
            ("rehashed_tamper", ["--cold-replay", str(tamper)], 1),
        ]
    ]


def main(argv: list[str] | None = None) -> int:
    """The direct entry owns deadlines and publishes only validated terminal bytes."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--private-fixture", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        e.progress("deliberate_error_rejected")
        return 1
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_failed")
        return int(not passed)
    if args.private_fixture and args.output.resolve().is_relative_to(
        (e.ROOT / "results").resolve()
    ):
        parser.error("--private-fixture requires output outside repository results")
    began = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        private = Path(tempfile.mkdtemp(dir=SCRATCH, prefix="invocation-"))
        os.environ["TMPDIR"] = str(private)
        raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, mode=0o700)
        frozen = [] if args.private_fixture else plan(private)
        atomic_json(raw / "validation_plan.json", dict(commands=frozen))
        work = e.measure(args.root, raw, private)
        work["code_config_hashes"].append(e.reference(raw / "validation_plan.json"))
        os.environ["CARNOT_8379_EXTENSION"] = work["extension"]["path"] or ""
        receipts = work["extension"]["receipts"][:]
        for index, command in enumerate(frozen):
            e.progress("validation", index, len(frozen) - index)
            receipts.append(
                reuse(command)
                if "receipt_reference" in command
                else child(
                    command["name"],
                    command["argv"],
                    raw / "validation",
                    deadline=command["deadline_s"],
                    scope=command["scope"],
                )
            )
        if args.private_fixture:
            receipts.append(dict(name="private_execution_control_only", passed=True, scope="owned"))
        for name in ("coverage.ini", "coverage.json", ".coverage"):
            source = Path(os.environ.get("CARNOT_8379_COVERAGE_ARTIFACT_DIR", str(private))) / name
            if source.is_file():
                target = raw / name
                target.write_bytes(source.read_bytes())
                work["code_config_hashes"].append(e.reference(target))
        atomic_json(raw / "measurement.json", work)
        candidate = private / (e.NAME + ".json")
        atomic_json(candidate, e.build(work, receipts))
        receipts.extend(controls(candidate, raw))
        initial = validate(candidate, raw / "initial_terminal")
        receipts.extend(initial["checks"])
        work["adversarial_findings"] = initial["adversarial"]["findings"]
        work["duration_s"] = time.monotonic() - began
        work["phase_spans"].extend(
            dict(phase=r["name"], duration_s=r.get("duration_s", 0)) for r in receipts
        )
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, receipts)
        publication = publish_primary(args.output, value, lambda p: validate(p, raw / "terminal"))
        atomic_json(Path(work["terminal_validation_sidecar_path"]), publication)
        e.progress("published_" + value["verdict_class"], 1, 0)
        return int(not value["required_checks_passed"])
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        e.progress("owned_failure_" + type(error).__name__ + "_" + str(error))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
