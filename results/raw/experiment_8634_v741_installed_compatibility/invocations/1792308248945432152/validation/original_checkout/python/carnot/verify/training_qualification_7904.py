"""Qualify fitting mechanics without claiming natural benefit (REQ-VERIFY-7904-V686)."""

from __future__ import annotations

from dataclasses import asdict, replace
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)
from carnot.verify import energy_fit_7894 as custody
from carnot.verify import natural_training, training_runtime

ROOT = Path(__file__).resolve().parents[3]
MODULE = "python/carnot/verify/training_qualification_7904.py"
CLI = "scripts/experiments/experiment_7904_v686_training_qualification.py"
OWNED = (MODULE, CLI)
NUMERICAL_MODULES = (
    "python/carnot/verify/natural_training.py",
    "python/carnot/verify/training_runtime.py",
    "python/carnot/verify/evidence_views.py",
    "python/carnot/verify/natural_predicates.py",
    "python/carnot/verify/source_alignment.py",
)
DEPENDENCIES = (*OWNED, *NUMERICAL_MODULES, "python/carnot/verify/energy_fit_7894.py")
TESTS = (
    "tests/python/test_training_qualification_7904.py",
    "tests/python/test_energy_fit_7894.py",
    "tests/python/test_natural_runtime_7853.py",
    "tests/python/test_natural_runtime_7867.py",
    "tests/python/test_source_boundary_7852.py",
)
INCLUDES = ",".join("*/" + name for name in OWNED)
START = time.monotonic()


def progress(phase: str, units: int = 0) -> None:
    """Show real elapsed work so a quiet numerical compiler stays observable."""
    print(
        f"[exp7904] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed={units}",
        flush=True,
    )


def dependency_hashes() -> dict[str, str]:
    """Authenticate numerical code and its public feature builders, not just a wrapper."""
    return {name: sha256_file(ROOT / name) for name in DEPENDENCIES}


def make_fixture(raw: Path, overlap: bool = False) -> Path:
    """Use separate byte and evaluator projections to exercise actual custody code."""
    raw.mkdir(parents=True, exist_ok=True)
    public, evaluator = [], []
    for role, number in (("fit", 12), ("tune", 14), ("evaluation", 16)):
        source = f"Lumen has {12 if overlap else number} apples."
        answer = f"Lumen has {number + 1} apples."
        public.append(
            {
                "family_id": role,
                "source_bytes": source.encode().hex(),
                "answer_bytes": answer.encode().hex(),
            }
        )
        evaluator.append(
            {
                "family_id": role,
                "role": role,
                "human_label": 1,
                "observed": 1,
                "annotation_byte_offsets": [[0, len(answer)]],
            }
        )
    artifact: dict[str, Any] = {
        "experiment_id": 7892,
        "source_boundary_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "circular_positive",
    }
    for key, rows in (("public_shards", public), ("evaluator_shards", evaluator)):
        path = raw / f"{key}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        artifact[key] = [{"path": str(path.resolve()), "sha256": sha256_file(path)}]
    upstream = raw / "upstream.json"
    atomic_json(upstream, artifact)
    return upstream.resolve()


def checkpoint_identity(upstream: Path, arm: str, seed: int, epochs: int) -> dict[str, Any]:
    """A second path is a second authority even when its current bytes happen to match."""
    records, sources = custody.load_records(upstream.resolve())
    return {
        "upstream_path": str(upstream.resolve()),
        "sources": sources,
        "cohort_sha256": canonical_hash(
            [{k: v for k, v in row.items() if k not in {"source", "answer"}} for row in records]
        ),
        "dependencies": dependency_hashes(),
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": natural_training.LEARNING_RATE,
        "format": "carnot-current-head-v686-1",
    }


def check_compatible(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject changed training authority before loading any saved numerical head."""
    if saved != current:
        raise ValueError("checkpoint dependency mismatch")


def mutation_rows(identity: dict[str, Any]) -> list[dict[str, Any]]:
    """Record actual reader rejections instead of deriving them from a passing test name."""
    rows = []
    for field in (*DEPENDENCIES, "upstream_path", "epochs", "arm", "seed", "format"):
        changed = json.loads(json.dumps(identity))
        if field in DEPENDENCIES:
            changed["dependencies"][field] = "sha256:changed"
        else:
            changed[field] = "changed"
        observed = None
        try:
            check_compatible(identity, changed)
        except ValueError as exc:
            observed = str(exc)
        rows.append(
            {
                "mutation": field,
                "rejected": observed is not None,
                "observed": observed,
                "original_identity_sha256": canonical_hash(identity),
                "changed_identity_sha256": canonical_hash(changed),
            }
        )
    return rows


def base(identity: int, date: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep terminal external blocks complete and keep fixture claims circular."""
    result: dict[str, Any] = {
        "experiment_id": identity,
        "task_id": "exp7904-training-qualification",
        "milestone": "2026.09.686",
        "run_date": date,
        "honest_verdict": "complete_circular_positive_training_fixture",
        "verdict_class": "circular_positive",
        "flagged_adversarial": False,
        "gate_check_summary": [],
        "rows": [],
        "sample_size_budget": {
            key: 0
            for key in (
                "intended",
                "eligible",
                "started",
                "completed",
                "failed",
                "censored",
                "excluded",
                "independent",
            )
        },
        "acceptance_gate_results": {
            "validity": True,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": time.monotonic() - START,
        "phase_spans": [],
        "random_seed": 67801,
        "reproducibility_checksum": canonical_hash(
            {"code": dependency_hashes(), "sources": sources}
        ),
        "source_artifact_hashes": sources,
        "preconditions_checked": {"failed": []},
        "resolved_imports": {
            "carnot.verify." + Path(name).stem: str(
                Path(importlib.import_module("carnot.verify." + Path(name).stem).__file__).resolve()
            )
            for name in (*NUMERICAL_MODULES, MODULE)
        },
        "validation_receipts": [],
        "validation_command_manifest_path": None,
        "observed_child_commands": [],
        "historical_required_failures": [],
        "repository_health": {"affects_required_checks": False},
        "verifier_is_oracle": True,
        "claim_scope": "private_fixture_oracle_agreement",
        "inference_substrate": "deterministic_cpu",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": None,
        "model_invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        "trained_head_specs": [],
        "training_runtime_ready_score": 0,
        "validation_scope_sha256": None,
        "training_dependency_hashes": dependency_hashes(),
        "checkpoint_mutation_rows": [],
        "fixture_prediction_rows": [],
        "coverage_statement_counts": {},
        "methodology": "Fresh private CPU head fitting, byte-bound checkpoint replay and required current callable validation. Fixture agreement measures mechanics only. Natural fitting belongs to Exp7906.",
    }
    result["field_principles"] = {
        key: "Preserve actual producer identity, primitive evidence and its scope; never infer independent benefit."
        for key in result
    }
    result["field_principles"]["acceptance_gate_results"] = {
        "validity": "Owned checks and exact terminal bytes pass.",
        "readiness": "Current runtime and full dependency compatibility qualify.",
        **{
            key: "Unmeasured natural benefit remains null."
            for key in ("probability_quality", "decision_benefit", "retention", "efficiency")
        },
    }
    return result


def fit_score(
    upstream: Path,
    output: Path,
    raw: Path,
    identity: int,
    date: str,
    resume: bool = False,
    deadline_s: float = 90,
) -> dict[str, Any]:
    """Fit fresh current heads; compatibility only authorizes an explicit same-run resume."""
    began = time.monotonic()
    progress("resolve_inputs")
    records, sources = custody.load_records(upstream.resolve())
    fit_rows = [row for row in records if row["role"] == "fit"]
    tune_rows = [row for row in records if row["role"] == "tune"]
    if {r["source_cluster_id"] for r in fit_rows} & {r["source_cluster_id"] for r in tune_rows}:
        raise ValueError("fit tune source group overlap")
    current = checkpoint_identity(upstream, "local_set", 67801, 1)
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "checkpoint_identity.json", current)
    checkpoint = raw / (canonical_hash(current).split(":")[1] + ".head.json")
    result = base(identity, date, sources)
    result["preconditions_checked"] = {
        "failed": [],
        "fit_tune_source_groups_disjoint": True,
        "upstream_path": str(upstream.resolve()),
    }
    if deadline_s <= 0:
        raise TimeoutError("owned fitting deadline")
    progress("before_head_fit")
    if resume and (raw / "checkpoint_manifest.json").is_file():
        saved = json.loads((raw / "checkpoint_manifest.json").read_text())
        check_compatible(saved["identity"], current)
        if sha256_file(checkpoint) != saved["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes mismatch")
        head = training_runtime.load(checkpoint)
    else:
        head = natural_training.fit(fit_rows, tune_rows, "local_set", 67801, 0.01, 1)
        training_runtime.save(checkpoint, head)
    if time.monotonic() - began > deadline_s:
        raise TimeoutError("owned fitting deadline")
    progress("after_head_fit", 1)
    atomic_json(
        raw / "checkpoint_manifest.json",
        {
            "identity": current,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
        },
    )
    predictions = natural_training.predict(head, records)
    rows = [
        {
            "family_id": record["id"],
            "unit": "fixture_family",
            "role": record["role"],
            "arm": "local_set",
            "seed": 67801,
            "label": record["label"],
            **prediction,
        }
        for record, prediction in zip(records, predictions, strict=True)
    ]
    primitive = raw / "predictions.jsonl"
    primitive.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    result.update(
        {
            "rows": rows,
            "fixture_prediction_rows": rows,
            "prediction_rows_path": str(primitive.resolve()),
            "prediction_rows_sha256": sha256_file(primitive),
            "checkpoint_manifest_path": str((raw / "checkpoint_manifest.json").resolve()),
            "checkpoint_manifest_sha256": sha256_file(raw / "checkpoint_manifest.json"),
            "checkpoint_identity": current,
            "trained_head_specs": [
                {
                    "arm": "local_set",
                    "seed": 67801,
                    "parameter_count": head["parameter_count"],
                    "fresh_current_fit": not resume,
                }
            ],
            "duration_s": time.monotonic() - START,
            "phase_spans": [{"phase": "private_fit_score", "duration_s": time.monotonic() - began}],
            "sample_size_budget": {
                **{key: len(rows) for key in ("intended", "eligible", "started", "completed")},
                **{key: 0 for key in ("failed", "censored", "excluded", "independent")},
                "unit": "fixture_family",
                "by_role": {
                    role: {
                        "unit": "fixture_family",
                        "intended": 1,
                        "eligible": 1,
                        "started": 1,
                        "completed": 1,
                        "failed": 0,
                        "censored": 0,
                        "excluded": 0,
                        "independent": 0,
                    }
                    for role in ("fit", "tune", "evaluation")
                },
            },
        }
    )
    atomic_json(output, result)
    return result


def replay(path: Path) -> dict[str, Any]:
    """Cold reduction verifies rows, checkpoint bytes and current training authority."""
    result = json.loads(path.read_text())
    primitive = Path(result["prediction_rows_path"])
    rows = [json.loads(line) for line in primitive.read_text().splitlines() if line]
    if (
        sha256_file(primitive) != result["prediction_rows_sha256"]
        or rows != result["fixture_prediction_rows"]
    ):
        raise ValueError("prediction replay mismatch")
    manifest_path = Path(result["checkpoint_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(manifest_path) != result["checkpoint_manifest_sha256"]:
        raise ValueError("checkpoint manifest mismatch")
    saved = manifest["identity"]
    check_compatible(
        saved,
        checkpoint_identity(
            Path(saved["upstream_path"]), saved["arm"], saved["seed"], saved["epochs"]
        ),
    )
    checkpoint = Path(manifest["checkpoint_path"])
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes mismatch")
    records, _ = custody.load_records(Path(saved["upstream_path"]))
    predictions = natural_training.predict(training_runtime.load(checkpoint), records)
    if predictions != [
        {key: row[key] for key in prediction}
        for row, prediction in zip(rows, predictions, strict=True)
    ]:
        raise ValueError("cold prediction mismatch")
    if result["sample_size_budget"]["completed"] != len(rows):
        raise ValueError("primitive denominator mismatch")
    progress("cold_replay_passed", len(rows))
    return result


def execute(
    commands: list[CommandSpec],
    raw: Path,
    heartbeat_s: float = 30,
    extra_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Seal each child's exited log by byte hash; never seal a still-running stream."""
    private = Path(tempfile.mkdtemp(prefix="carnot7904-child-"))
    receipts = run_commands(
        ROOT, commands, log_dir=private, heartbeat_s=heartbeat_s, extra_env=extra_env
    )
    for spec, row in zip(commands, receipts, strict=True):
        log = Path(row["log_path"])
        durable = raw / "logs" / (sha256_file(log).split(":")[1] + ".log")
        durable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, durable)
        row.update(
            {
                "log_path": str(durable.resolve()),
                "argv": list(spec.argv),
                "actual_exit": row["exit_code"],
                "expected_exit": 0,
                "deadline_s": spec.timeout_s,
                "expected_reason": "current affected command must pass",
                "measured_files": list(OWNED),
            }
        )
    return receipts


def readers(candidate: Path, raw: Path, recheck: Path | None = None) -> list[dict[str, Any]]:
    """Fresh repository readers inspect exact candidate bytes, including disqualified bytes."""
    commands = [
        CommandSpec(
            "adversarial",
            (sys.executable, "scripts/adversarial_verify.py", "--json", str(candidate)),
            "terminal",
            180,
        ),
        CommandSpec(
            "row_consistency",
            (sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal",
            180,
        ),
    ]
    if recheck:
        commands.append(
            CommandSpec("terminal_recheck", (sys.executable, str(recheck)), "terminal", 30)
        )
    receipts = execute(commands, raw)
    for row in receipts:
        row["candidate_sha256"] = sha256_file(candidate)
    return receipts


def disqualify(result: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    """Owned failures close readiness even when all numerical rows exist."""
    result.update(
        {
            "honest_verdict": "complete_disqualified_training_qualification",
            "verdict_class": "disqualified",
            "training_runtime_ready_score": 0,
        }
    )
    result["acceptance_gate_results"].update({"validity": False, "readiness": 0})
    result["gate_check_summary"] += failures


def publish(result: dict[str, Any], output: Path, raw: Path, recheck: Path | None = None) -> bool:
    """Publish only the last bytes read by both validators; receipts live outside those bytes."""
    candidate = raw / "terminal_candidate.json"
    result["terminal_validation_sidecar_path"] = str(output.resolve()) + ".validators.json"
    atomic_json(candidate, result)
    first = readers(candidate, raw)
    second = readers(candidate, raw, recheck)
    all_receipts = first + second
    if any(not row["passed"] for row in all_receipts):
        adversarial = [row for row in all_receipts if row["name"] == "adversarial"]
        result["flagged_adversarial"] = any(
            json.loads(Path(row["log_path"]).read_text())["flagged_count"] > 0
            for row in adversarial
        )
        disqualify(
            result,
            [
                {
                    "upstream_id": "owned_terminal_reader",
                    "artifact_path": str(candidate),
                    "artifact_sha256": row["candidate_sha256"],
                    "field": row["name"],
                    "op": "==",
                    "expected": 0,
                    "observed": row["actual_exit"],
                }
                for row in all_receipts
                if not row["passed"]
            ],
        )
        atomic_json(candidate, result)
        all_receipts += readers(candidate, raw)
    digest = sha256_file(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".checked")
    shutil.copyfile(candidate, temporary)
    temporary.replace(output)
    atomic_json(
        Path(result["terminal_validation_sidecar_path"]),
        {"candidate_sha256": digest, "receipts": all_receipts},
    )
    progress("checked_terminal_published", len(result["rows"]))
    return result["verdict_class"] not in {"disqualified", "blocked"} and all(
        row["passed"] for row in all_receipts
    )


def history() -> dict[str, Any]:
    """Reproduce the old failure from its sealed log without relabeling current source edits."""
    path = ROOT / "results/experiment_7894_v685_energy_fit.json"
    artifact = json.loads(path.read_text())
    failed = [row for row in artifact["validation_receipts"] if not row["passed"]]
    for row in failed:
        if sha256_file(Path(row["log_path"])) != row["log_sha256"]:
            raise custody.InputBlocked(
                [
                    custody.operand(
                        path,
                        "historical_required_log",
                        "hash_matches",
                        row["log_sha256"],
                        sha256_file(Path(row["log_path"])),
                    )
                ]
            )
    manifest = Path(artifact["validation_command_manifest_path"])
    spec = next(
        row
        for row in json.loads(manifest.read_text())["commands"]
        if row["name"] == "spec_coverage"
    )
    source = ROOT / "scripts/experiments/experiment_7894_v685_energy_fit.py"
    old_checkpoint_manifest = Path(artifact["checkpoint_manifest_path"])
    old_code = json.loads(old_checkpoint_manifest.read_text())["code_hashes"]
    return {
        "historical_required_failures": [
            *artifact["historical_required_failures"],
            *[{"experiment_id": 7894, **row} for row in failed],
        ],
        "repository_health": {
            "affects_required_checks": False,
            "historical_exp7894_verdict": artifact["honest_verdict"],
            "global_spec_backlog": 1142,
            "backlog_evidence": failed,
            "current_manifest_spec_argv": spec["argv"],
            "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "current_source_sha256": sha256_file(source),
            "historical_source_sha256": old_code[
                "scripts/experiments/experiment_7894_v685_energy_fit.py"
            ],
            "historical_checkpoint_manifest_sha256": sha256_file(old_checkpoint_manifest),
            "source_hash_comparisons": {
                name: {
                    "historical": value,
                    "current": sha256_file(ROOT / name),
                    "equal": value == sha256_file(ROOT / name),
                }
                for name, value in old_code.items()
            },
            "source_hash_comparison": "Historical artifact did not seal the transitive dependency closure; equality cannot be authenticated.",
            "retire_if_same_verdict": True,
        },
    }


def freeze(private: Path, raw: Path, date: str) -> tuple[Path, list[CommandSpec]]:
    """Register exact bounded argv and complete hashes before any numerical fitting."""
    coverage_file = private / ".coverage"
    commands = build_scoped_commands(
        ROOT, TESTS, (MODULE,), static_paths=(CLI,), basetemp=private, coverage_file=coverage_file
    )
    commands = [
        replace(
            row,
            timeout_s=600 if "pytest" in row.name or row.name == "changed_module_coverage" else 180,
        )
        for row in commands
    ]
    index = next(i for i, row in enumerate(commands) if row.name == "changed_module_coverage")
    commands[index] = CommandSpec(
        "changed_module_coverage",
        (
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            f"--data-file={coverage_file}",
            f"--include={INCLUDES}",
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={private / 'coverage'}",
            *TESTS,
            "-q",
        ),
        "units_and_real_cli",
        600,
    )
    commands.insert(
        index + 1,
        CommandSpec(
            "combine_coverage",
            (
                sys.executable,
                "-m",
                "coverage",
                "combine",
                f"--data-file={coverage_file}",
                str(private),
            ),
            "changed_source",
            60,
        ),
    )
    report_index = next(
        i for i, row in enumerate(commands) if row.name == "changed_module_coverage_report"
    )
    commands[report_index] = CommandSpec(
        "changed_module_coverage_report",
        (
            sys.executable,
            "-m",
            "coverage",
            "json",
            f"--data-file={coverage_file}",
            f"--include={INCLUDES}",
            "--fail-under=100",
            "-o",
            str(private / "coverage.json"),
        ),
        "changed_source",
        60,
    )
    mypy_index = next(i for i, row in enumerate(commands) if row.name == "changed_module_mypy")
    commands[mypy_index] = replace(
        commands[mypy_index],
        argv=(str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED),
    )
    intervention = "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
    commands += [
        CommandSpec(
            "e2e_016_fixture",
            (
                sys.executable,
                intervention,
                "--date",
                date,
                "--fixture-e2e",
                str(private / "e2e016.json"),
            ),
            "private_fixture",
            180,
        ),
        CommandSpec(
            "e2e_016_replay",
            (
                sys.executable,
                intervention,
                "--date",
                date,
                "--cold-replay",
                str(private / "e2e016.json"),
            ),
            "cold_replay",
            180,
        ),
    ]
    manifest = raw / "validation_command_manifest.json"
    atomic_json(
        manifest,
        {
            "commands": [
                {
                    **asdict(row),
                    "expected_exit": 0,
                    "expected_reason": "required affected check",
                    "deadline_s": row.timeout_s,
                }
                for row in commands
            ],
            "changed_modules": list(OWNED),
            "affected_dependency_closure": dependency_hashes(),
            "tests": list(TESTS),
            "coverage_includes": INCLUDES,
            "full_repository_health_is_separate": True,
        },
    )
    return manifest, commands


def qualify(upstream: Path, output: Path, raw: Path, date: str, private: Path | None = None) -> int:
    """Qualify the actual private fitting CLI before assigning runtime readiness."""
    records, sources = custody.load_records(upstream.resolve())
    historical = history()
    private = private or Path(tempfile.mkdtemp(prefix="carnot7904-qualification-"))
    fixture = make_fixture(private / "fixture-input")
    manifest, commands = freeze(private, raw, date)
    fixture_commands = [
        CommandSpec(
            "fixture_cli_success",
            (
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                f"--include={INCLUDES}",
                CLI,
                "--date",
                date,
                "--fixture",
                "--upstream",
                str(fixture),
                "--output",
                str(private / "fixture.json"),
                "--raw-root",
                str(private / "fixture-raw"),
            ),
            "private_cpu",
            180,
        ),
        CommandSpec(
            "fixture_cli_replay",
            (
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                f"--include={INCLUDES}",
                CLI,
                "--date",
                date,
                "--cold-replay",
                str(private / "fixture.json"),
            ),
            "private_cpu",
            180,
        ),
    ]
    frozen = json.loads(manifest.read_text())
    frozen["commands"] = [
        {
            **asdict(row),
            "expected_exit": 0,
            "expected_reason": "real private CLI qualification",
            "deadline_s": row.timeout_s,
        }
        for row in fixture_commands
    ] + frozen["commands"]
    frozen["source_artifact_hashes"] = sources
    frozen["fixture_identity"] = checkpoint_identity(fixture, "local_set", 67801, 1)
    atomic_json(manifest, frozen)
    receipts = execute(
        fixture_commands + commands,
        raw,
        extra_env={"CARNOT7904_COVERAGE": "1", "COVERAGE_FILE": str(private / ".coverage")},
    )
    if not (private / "fixture.json").is_file():
        result = base(7904, date, sources)
        result.update(historical)
        result.update(
            {
                "validation_receipts": receipts,
                "observed_child_commands": [row["argv"] for row in receipts],
                "validation_command_manifest_path": str(manifest),
                "validation_scope_sha256": sha256_file(manifest),
            }
        )
        disqualify(
            result,
            [
                {
                    "upstream_id": "owned_fixture_cli",
                    "artifact_path": str(private / "fixture.json"),
                    "artifact_sha256": None,
                    "field": "terminal_fixture",
                    "op": "exists",
                    "expected": True,
                    "observed": None,
                }
            ],
        )
        publish(result, output, raw)
        return 2
    result = json.loads((private / "fixture.json").read_text())
    sealed = raw / "sealed_fixture"
    shutil.copytree(private / "fixture-raw", sealed, dirs_exist_ok=True)
    checkpoint_manifest = json.loads(Path(result["checkpoint_manifest_path"]).read_text())
    checkpoint_manifest["checkpoint_path"] = str(
        (sealed / Path(checkpoint_manifest["checkpoint_path"]).name).resolve()
    )
    atomic_json(sealed / "checkpoint_manifest.json", checkpoint_manifest)
    result.update(
        {
            "prediction_rows_path": str((sealed / "predictions.jsonl").resolve()),
            "checkpoint_manifest_path": str((sealed / "checkpoint_manifest.json").resolve()),
            "checkpoint_manifest_sha256": sha256_file(sealed / "checkpoint_manifest.json"),
        }
    )
    coverage_path = private / "coverage.json"
    coverage = (
        json.loads(coverage_path.read_text()).get("files", {}) if coverage_path.is_file() else {}
    )
    counts = {name: value["summary"] for name, value in coverage.items()}
    complete_coverage = all(
        name in counts and counts[name]["num_statements"] > 0 and counts[name]["missing_lines"] == 0
        for name in OWNED
    )
    result.update(historical)
    health_path = raw / "repository_health/receipt.json"
    if health_path.is_file():
        result["repository_health"]["full_suite"] = json.loads(health_path.read_text())
        result["repository_health"]["full_suite_sha256"] = sha256_file(health_path)
    result.update(
        {
            "validation_receipts": receipts,
            "observed_child_commands": [row["argv"] for row in receipts],
            "validation_command_manifest_path": str(manifest.resolve()),
            "validation_scope_sha256": sha256_file(manifest),
            "coverage_statement_counts": counts,
            "source_artifact_hashes": [*result["source_artifact_hashes"], *sources],
            "preconditions_checked": {
                "failed": [],
                "upstream_records_authenticated": len(records),
                "natural_fitting_deferred_to": 7906,
            },
            "duration_s": time.monotonic() - START,
            "training_runtime_ready_score": 1,
            "checkpoint_mutation_rows": mutation_rows(result["checkpoint_identity"]),
            "reproducibility_checksum": canonical_hash({"scope": frozen, "sources": sources}),
        }
    )
    result["acceptance_gate_results"]["readiness"] = 1
    failures = [
        {
            "upstream_id": "owned_affected_validation",
            "artifact_path": str(manifest),
            "artifact_sha256": sha256_file(manifest),
            "field": row["name"],
            "op": "==",
            "expected": 0,
            "observed": row["actual_exit"],
        }
        for row in receipts
        if not row["passed"]
    ]
    if not complete_coverage:
        failures.append(
            {
                "upstream_id": "owned_coverage",
                "artifact_path": str(coverage_path),
                "artifact_sha256": sha256_file(coverage_path) if coverage_path.is_file() else None,
                "field": "changed_statement_coverage",
                "op": "==",
                "expected": "nonempty 100% each changed file",
                "observed": counts,
            }
        )
    if failures:
        disqualify(result, failures)
    atomic_json(raw / "qualified_candidate.json", result)
    return 0 if publish(result, output, raw) else 2
