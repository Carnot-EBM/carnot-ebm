#!/usr/bin/env python3
"""REQ-REPORT-7848: run and validate the exposed-cohort length control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.length_shortcut import (
    CustodyError,
    authenticate,
    cold_replay,
    energy_gate,
    run_baseline,
)


ROOT = Path(__file__).resolve().parents[2]
TEST = "tests/python/test_experiment_7848_v681_length_shortcut.py"
MODULE = "python/carnot/reporting/length_shortcut.py"
CLI = "scripts/experiments/experiment_7848_v681_length_shortcut.py"
REQUIRED = (
    "worktree_imports",
    "affected_pytest",
    "changed_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec",
    "cli_e2e",
    "cold_replay",
    "adversarial_verify",
    "strict_rows",
)


def progress(phase: str, start: float, done: int = 0, total: int = 1) -> None:
    """Print a flushed phase boundary with elapsed time and completed units."""
    print(
        f"[exp7848] {phase} elapsed_s={time.monotonic() - start:.3f} completed={done}/{total}",
        flush=True,
    )


def science(output: Path, date: str) -> dict[str, Any]:
    """Produce a complete independent baseline without recursive validation."""
    start = time.monotonic()
    start_ns = time.monotonic_ns()
    progress("start", start)
    output.mkdir(parents=True, exist_ok=True)
    phase_spans = []
    before = time.monotonic()
    progress("preconditions_begin", start)
    try:
        custody = authenticate(ROOT)
        energy = energy_gate(ROOT)
        progress("preconditions_end", start, 1)
    except CustodyError as exc:
        progress("preconditions_blocked", start, 1)
        blocked = dict(
            experiment_id=7848,
            task_id="exp7848-length-shortcut",
            milestone="2026.09.681",
            run_date=date,
            honest_verdict="complete_blocked_required_exp7810_evidence",
            verdict_class="blocked",
            flagged_adversarial=True,
            gate_check_summary=exc.failures,
            rows=[],
            sample_size_budget=dict(
                intended=64,
                eligible=0,
                started=0,
                completed=0,
                censored=0,
                excluded=64,
                independent_n=0,
            ),
            length_control_ready_score=0,
            permutation_rows=[],
            inference_substrate="verifier_ensemble_against_cached_candidates",
            inference_substrate_class="no_model_load",
            planned_inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_specs=[],
            model_invocation_counts={},
            acceptance_gate_results=dict(
                validity=False,
                readiness=0,
                probability_quality=None,
                decision_benefit=None,
                retention=None,
                efficiency=None,
            ),
            duration_s=time.monotonic() - start,
            phase_spans=[],
            random_seed=7848,
            source_artifact_hashes=[],
            preconditions_checked=[],
            validation_receipts=[],
            observed_child_commands=[],
            validation_command_manifest_path=None,
            repository_health=None,
            verifier_is_oracle=False,
            claim_scope="exposed_development_only",
            field_principles={},
            length_model=None,
            strata_definition=None,
            baseline_rows=[],
            semantic_interpretation_gate=None,
            reproducibility_checksum=canonical_hash(exc.failures),
        )
        atomic_json(output / "candidate.json", blocked)
        return blocked
    phase_spans.append(
        dict(phase="preconditions", start_s=before - start, end_s=time.monotonic() - start)
    )
    progress("fit_begin", start)
    before = time.monotonic()
    result = run_baseline(custody, output, seed=7848)
    phase_spans.append(
        dict(phase="fit_tune_evaluate", start_s=before - start, end_s=time.monotonic() - start)
    )
    progress("fit_end", start, 64, 64)
    progress("cold_replay_begin", start)
    replay = cold_replay(result, custody, output)
    if not replay["passed"]:
        raise ValueError(f"cold_replay_failed:{replay}")
    progress("cold_replay_end", start, 64, 64)
    duration = time.monotonic() - start
    receipt = build_current_work_receipt(
        run_id=f"exp7848-{result['config_sha256'][7:23]}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"public_feature_count": 3, "model_loads": 0},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=phase_spans,
    )
    source_hashes = [
        *custody["source_artifact_hashes"],
        dict(
            upstream_id="exp7840",
            path=energy.get("source_path"),
            sha256=energy.get("source_sha256"),
            eligibility="blocked_optional_energy",
            date="20260929",
        ),
    ]
    artifact = dict(
        experiment_id=7848,
        task_id="exp7848-length-shortcut",
        milestone="2026.09.681",
        run_date=date,
        honest_verdict="complete_null_length_control_energy_unavailable",
        verdict_class="null",
        flagged_adversarial=any(s.get("flagged_adversarial", False) for s in (energy,)),
        gate_check_summary=[],
        rows=result["rows"],
        sample_size_budget=result["sample_size_budget"],
        acceptance_gate_results=dict(
            validity=True,
            readiness=1,
            probability_quality=result["bootstrap"]["brier_gain_over_prevalence"]["mean"],
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        duration_s=duration,
        phase_spans=phase_spans,
        random_seed=7848,
        reproducibility_checksum=canonical_hash(
            dict(
                config=result["config_sha256"],
                sources=[s["sha256"] for s in source_hashes if s.get("sha256")],
                code=[sha256_file(ROOT / path) for path in (MODULE, CLI)],
            )
        ),
        source_artifact_hashes=source_hashes,
        preconditions_checked=custody["preconditions_checked"],
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        repository_health=None,
        verifier_is_oracle=False,
        claim_scope="exposed_human_labeled_development_only; no fresh generalization",
        field_principles={
            "experiment_id": "Integer experiment number differs from task slug.",
            "rows": "One intact human-labelled family is one independent unit.",
            "length_model": "Public UTF-8 byte lengths do not prove source semantics.",
            "gate_check_summary": "Required missing evidence ends blocked with exact operands.",
            "acceptance_gate_results": "Optional energy interpretation cannot override baseline validity.",
        },
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        planned_inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=receipt["invocation_counts"],
        current_work_receipt=receipt,
        length_control_ready_score=1,
        length_model=result["length_model"],
        strata_definition=result["strata_definition"],
        strata_metrics=result["strata_metrics"],
        baseline_rows=result["baseline_rows"],
        bootstrap=result["bootstrap"],
        permutation_rows=energy["permutation_rows"],
        semantic_interpretation_gate=dict(
            passed=False,
            branch="blocked_external_exp7840",
            failures=energy["failures"],
            required_cost_gain=0.02,
            required_cost_lower95_positive=True,
            required_brier_lower95_positive=True,
            required_source_erased_advantage=True,
        ),
        model_checkpoint_path=result["model_checkpoint_path"],
        model_checkpoint_sha256=result["model_checkpoint_sha256"],
        prediction_path=result["prediction_path"],
        prediction_seal_sha256=result["prediction_seal_sha256"],
        config_sha256=result["config_sha256"],
        cold_replay=replay,
        inapplicable_e2e={
            f"E2E-{i:03d}": "Unrelated model, hardware or ARC path; reporting CLI is checked separately."
            for i in range(1, 15)
        },
    )
    atomic_json(output / "candidate.json", artifact)
    progress("candidate_written", start, 64, 64)
    return artifact


def commands(output: Path) -> list[dict[str, Any]]:
    """Freeze explicit argv, scope, deadlines, and classification before children run."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    coverage = str(ROOT / ".venv/bin/coverage")
    unit_data = output / ".coverage.unit"
    cli_data = output / ".coverage.cli"
    combined = output / ".coverage.combined"
    private = output / "cli_private"
    candidate = output / "candidate.json"
    imports = (
        "import importlib,json,pathlib; names=('carnot.reporting.length_shortcut',"
        "'carnot.reporting.current_work_receipt'); r={n:str(pathlib.Path("
        "importlib.import_module(n).__file__).resolve()) for n in names}; "
        "print(json.dumps({'resolved_imports':r},sort_keys=True)); "
        f"assert all(p.startswith({str(ROOT / 'python')!r}) for p in r.values())"
    )
    coverage_report = (
        "from coverage import Coverage, CoverageData; "
        f"files={[str(unit_data), str(cli_data)]!r}; "
        "d=CoverageData(); "
        "[(lambda x:(x.read(),d.update(x)))(CoverageData(basename=f)) for f in files]; "
        f"c=Coverage(data_file={str(combined)!r}); c.get_data().update(d); c.save(); "
        f"p=c.report(include={[str(ROOT / MODULE), str(ROOT / CLI)]!r},show_missing=True); "
        "print({'statement_coverage_percent':p,'data_files':files}); "
        "raise SystemExit(0 if p==100.0 else 1)"
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    entries = [
        ("worktree_imports", [py, "-u", "-c", imports], 30, "required"),
        (
            "affected_pytest",
            [
                coverage,
                "run",
                f"--data-file={unit_data}",
                f"--include={ROOT / MODULE},{ROOT / CLI}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={output / 'pytest'}",
                TEST,
                "-q",
            ],
            120,
            "required",
        ),
        (
            "cli_e2e",
            [
                coverage,
                "run",
                f"--data-file={cli_data}",
                f"--include={ROOT / MODULE},{ROOT / CLI}",
                CLI,
                "--date",
                "20260929",
                "--output-root",
                str(private),
                "--science-only",
            ],
            120,
            "required",
        ),
        ("changed_coverage", [py, "-u", "-c", coverage_report], 30, "required"),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", MODULE, CLI, TEST], 60, "required"),
        (
            "ruff_format",
            [str(ROOT / ".venv/bin/ruff"), "format", "--check", MODULE, CLI, TEST],
            60,
            "required",
        ),
        ("mypy", [str(ROOT / ".venv/bin/mypy"), "--strict", MODULE, CLI], 120, "required"),
        ("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", TEST], 30, "required"),
        ("cold_replay", [py, "-u", CLI, "--cold-replay", str(candidate)], 60, "required"),
        (
            "adversarial_verify",
            [py, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)],
            120,
            "required",
        ),
        (
            "strict_rows",
            [py, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            60,
            "required",
        ),
        ("repository_health_180s", [pytest, "tests/python", "-q"], 180, "diagnostic"),
    ]
    return [
        dict(
            name=name,
            argv=argv,
            deadline_s=deadline,
            classification=classification,
            cwd=str(ROOT),
            env={"PYTHONPATH": "python:.", "JAX_PLATFORMS": "cpu"},
        )
        for name, argv, deadline, classification in entries
    ]


def run_child(spec: dict[str, Any], output: Path, start: float) -> dict[str, Any]:
    """Supervise only the owned process; seal its closed output by content hash."""
    name = spec["name"]
    progress(f"before_subprocess:{name}", start)
    began = time.monotonic()
    env = dict(os.environ)
    env.update(spec["env"])
    child = subprocess.Popen(
        spec["argv"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    chunks = []
    timed_out = False
    while True:
        remaining = spec["deadline_s"] - (time.monotonic() - began)
        if remaining <= 0:
            os.killpg(child.pid, signal.SIGKILL)
            timed_out = True
            tail, _ = child.communicate(timeout=5)
            chunks.append(tail)
            break
        try:
            tail, _ = child.communicate(timeout=min(55, remaining))
            chunks.append(tail)
            break
        except subprocess.TimeoutExpired:
            progress(f"heartbeat:{name}", start)
    payload = "".join(chunks)
    log_hash = "sha256:" + __import__("hashlib").sha256(payload.encode()).hexdigest()
    log = output / "validation_logs" / name / f"{log_hash[7:]}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text(payload, encoding="utf-8")
    if sha256_file(log) != log_hash:
        raise ValueError(f"sealed_log_mismatch:{name}")
    receipt = dict(
        name=name,
        command_argv=spec["argv"],
        classification=spec["classification"],
        deadline_s=spec["deadline_s"],
        exit_code=child.returncode,
        timed_out=timed_out,
        passed=child.returncode == 0 and not timed_out,
        duration_s=time.monotonic() - began,
        log_path=str(log),
        log_sha256=log_hash,
        output_tail=payload[-2000:],
    )
    if name == "worktree_imports":
        try:
            receipt["resolved_imports"] = json.loads(payload.strip())["resolved_imports"]
        except (ValueError, KeyError):
            receipt["resolved_imports"] = {}
        receipt["passed"] = receipt["passed"] and bool(receipt["resolved_imports"])
    progress(f"after_subprocess:{name}:exit={child.returncode}", start, 1)
    return receipt


def _historical_failures() -> list[dict[str, Any]]:
    """Carry prior blocked measurements as immutable references, never current checks."""
    ids = (7731, 7744, 7758, 7772, 7786, 7799, 7813, 7827)
    rows = []
    for number in ids:
        paths = sorted((ROOT / "results").glob(f"experiment_{number}*.json"))
        rows.append(
            dict(
                experiment_id=number,
                artifact_paths=[dict(path=str(path), sha256=sha256_file(path)) for path in paths],
                status="historical_unresolved",
                affects_current_required_checks=False,
            )
        )
    return rows


def validate(candidate: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run the frozen affected checks, then classify the terminal candidate."""
    start = time.monotonic()
    frozen = commands(output)
    manifest_path = output / "validation_command_manifest.json"
    atomic_json(
        manifest_path,
        dict(
            schema="exp7848_validation_v1",
            closure=[MODULE, CLI, TEST],
            commands=frozen,
            required=list(REQUIRED),
            repository_health_name="repository_health_180s",
        ),
    )
    candidate["validation_command_manifest_path"] = str(manifest_path)
    candidate["validation_command_manifest_sha256"] = sha256_file(manifest_path)
    candidate["historical_failures"] = _historical_failures()
    atomic_json(output / "candidate.json", candidate)
    receipts = []
    for index, spec in enumerate(frozen):
        progress(f"validation_unit:{spec['name']}", start, index, len(frozen))
        receipt = run_child(spec, output, start)
        if spec["classification"] == "diagnostic":
            candidate["repository_health"] = dict(
                status="healthy" if receipt["passed"] else "failed_health",
                receipt=receipt,
                historical_failures=candidate["historical_failures"],
            )
        else:
            receipts.append(receipt)
    candidate["validation_receipts"] = receipts
    candidate["observed_child_commands"] = [r["command_argv"] for r in receipts]
    failures = [r["name"] for r in receipts if not r["passed"]]
    failures.extend(name for name in REQUIRED if name not in {r["name"] for r in receipts})
    candidate["required_validation_failures"] = failures
    adverse = next((r for r in receipts if r["name"] == "adversarial_verify"), None)
    if adverse is not None:
        try:
            report = json.loads(Path(adverse["log_path"]).read_text(encoding="utf-8"))
            candidate["flagged_adversarial"] = bool(
                report.get("flagged_count", 0)
                or report.get("flagged", False)
                or report.get("critical_flags", [])
                or report.get("high_flags", [])
            )
        except (ValueError, AttributeError):
            candidate["flagged_adversarial"] = not adverse["passed"]
    if failures or candidate["flagged_adversarial"]:
        candidate["honest_verdict"] = "complete_disqualified_required_validation"
        candidate["verdict_class"] = "disqualified"
        candidate["length_control_ready_score"] = 0
        candidate["acceptance_gate_results"]["validity"] = False
        candidate["acceptance_gate_results"]["readiness"] = 0
    candidate["duration_s"] += time.monotonic() - start
    candidate["phase_spans"].append(
        dict(
            phase="validation",
            start_s=candidate["duration_s"] - (time.monotonic() - start),
            end_s=candidate["duration_s"],
        )
    )
    final_path = ROOT / "results/experiment_7848_v681_length_shortcut.json"
    atomic_json(final_path, candidate)
    progress("final_artifact_written", start, len(frozen), len(frozen))
    return candidate


def main() -> int:
    """Run science, private child replay, or the full task-owned validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260929")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--science-only", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args()
    if args.cold_replay is not None:
        candidate = json.loads(args.cold_replay.read_text(encoding="utf-8"))
        custody = authenticate(ROOT)
        output = Path(candidate["prediction_path"]).parent
        replay = cold_replay(candidate, custody, output)
        for receipt in candidate.get("validation_receipts", []):
            path = Path(receipt["log_path"])
            if not path.is_file() or sha256_file(path) != receipt["log_sha256"]:
                replay = dict(passed=False, reason="validation_log_mismatch", path=str(path))
                break
        print(json.dumps(replay, sort_keys=True), flush=True)
        return 0 if replay["passed"] else 1
    output = args.output_root or ROOT / "results/raw/experiment_7848_v681_length_shortcut/current"
    candidate = science(output, args.date)
    if args.science_only:
        return 0
    if candidate["verdict_class"] == "blocked":
        atomic_json(ROOT / "results/experiment_7848_v681_length_shortcut.json", candidate)
        return 0
    validate(candidate, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
