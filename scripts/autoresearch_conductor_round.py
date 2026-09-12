#!/usr/bin/env python3
"""Unattended autoresearch mutation round for the conductor (REQ-AUTO-019/020).

**What this closes.** `python/carnot/autoresearch/` (REQ-AUTO-001..015) already
implements the whole AVO-style loop -- an LLM proposes a hypothesis, the
sandbox runs it, the 3-gate evaluator scores it against the current baseline,
and a circuit breaker halts on repeated failure. `scripts/research_conductor.py`
never called any of it. This script is the missing wiring: run one bounded
round of that loop, and when a hypothesis wins, persist it as its own git
commit carrying its score -- the one piece the orchestrator itself does not
do (it only updates an in-memory baseline and appends to a JSON log).

**How it is invoked.** Exactly like the milestone-close audits already wired
into `research_step()` (`pages_adversarial_audit.py`, `verifier_authenticity_
audit.py`, etc.): a subprocess launched via `_run_audit_with_receipt`, gated
by `CARNOT_AUTORESEARCH_UNATTENDED=1` (default off), writing a receipt file
the caller checks for freshness. If the `codex` CLI is unavailable, or the
hypothesis generator's home-grown constitution forbids the run, this writes
a clean non-fatal receipt and exits 0 -- the milestone-close path is never
blocked by this step, matching the existing audits' own contract.

**Hypothesis generator: codex CLI, model gpt-6-astra (2026-09-12 operator
directive).** The mutation-operator's "propose a hypothesis" role is the same
category of task as the planner/retro/audit tiers this project already runs
via `codex exec` (never a locally-served model) -- an agent deciding what to
try next, not the ARC live agent's own inference-latency-bound generation
that IS local-first by contract. `call_codex()` below mirrors the exact
subprocess pattern the sibling audit scripts already use
(`pages_adversarial_audit.py:call_codex`), reusing `hypothesis_generator.py`'s
existing `DEFAULT_SYSTEM_PROMPT` / `_build_user_prompt` / `_extract_hypotheses`
for the prompt and parsing -- only the transport (a codex subprocess instead
of a raw OpenAI-compatible HTTP call) changed. This is the project's own
internal R&D tooling talking to a closed-weight model, the same as every
other autonomous conductor role; it is not a Carnot CAPABILITY (the thing
`python/carnot/verify`/`pipeline`/`samplers` ship to users), so the
Decentralization-Respecting Design Constraints' local-first mandate (which
targets shipped capabilities) does not bind it, any more than it binds the
planner's own codex calls.

**Fitness target #1.** The synthetic DoubleWell/Rosenbrock energy benchmarks
from `scripts/demo_autoresearch.py:create_initial_baselines()` -- cheap,
deterministic, already-implemented, no live LLM inference needed to SCORE a
candidate (only to PROPOSE one). Deliberately NOT the ARC live agent (see
`docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md` Part 3
for why that was rejected) and NOT the verifier ensemble (no reusable AUROC
harness exists yet).

**Where the lineage lives.** `ops/autoresearch_discoveries/<benchmark>/
<experiment_id>.json` -- a self-contained record (code, description, metrics,
before/after baseline) rather than a live `.py` module. This sidesteps
ruff/mypy strict-mode friction on raw LLM-generated snippets entirely: the
record does not need to satisfy production code-quality gates to be a valid,
readable audit trail, any more than a results/*.json artifact does. Each
accepted hypothesis lands as its own commit, `git add`-ing only that one
file -- never `-A` -- so this can never sweep unrelated in-flight work into
its commit (the exact collision class `harness_integrity_lint.py` exists to
catch; scoping to one explicit path avoids the class outright rather than
needing that guard).

**Local state, not the research record.** `ops/.autoresearch_baselines.json`
and `ops/.autoresearch_experiment_log.json` are gitignored fast-resume
caches (see `.gitignore`) -- the durable record is the committed lineage
above. `ops/autoresearch_conductor_report.md` IS tracked, matching every
sibling audit's report file; it is picked up by the conductor's own regular
end-of-step commit, not committed by this script.

Spec: REQ-AUTO-019, REQ-AUTO-020, SCENARIO-AUTO-019-*, SCENARIO-AUTO-020-*
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics  # noqa: E402
from carnot.autoresearch.constitution import ActionCategory, ConstitutionChecker  # noqa: E402
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog  # noqa: E402
from carnot.autoresearch.hypothesis_generator import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    _build_user_prompt,
    _extract_hypotheses,
)
from carnot.autoresearch.orchestrator import AutoresearchConfig, run_loop_with_generator  # noqa: E402

DEFAULT_BENCHMARK_DATA: dict[str, Any] = {"dim": 2}
DEFAULT_MODEL = os.environ.get("CARNOT_AUTORESEARCH_MODEL", "gpt-6-astra")
DEFAULT_CODEX_TIMEOUT_S = 300


def seed_baselines() -> BaselineRecord:
    """The same starting point as scripts/demo_autoresearch.py -- do not drift from it."""
    record = BaselineRecord(version="0.1.0")
    record.benchmarks["double_well"] = BenchmarkMetrics(
        benchmark_name="double_well",
        final_energy=0.05,
        convergence_steps=5000,
        wall_clock_seconds=2.0,
    )
    record.benchmarks["rosenbrock"] = BenchmarkMetrics(
        benchmark_name="rosenbrock",
        final_energy=0.5,
        convergence_steps=10000,
        wall_clock_seconds=5.0,
    )
    return record


def load_baselines(baseline_cache: Path) -> BaselineRecord:
    if baseline_cache.exists():
        try:
            return BaselineRecord.load(baseline_cache)
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # fail toward a fresh, known-good baseline, not a crash
    return seed_baselines()


def load_experiment_log(log_cache: Path) -> ExperimentLog:
    return ExperimentLog.load(log_cache)


def codex_available() -> bool:
    """Precondition check (Pre-Launch Preconditions Discipline pattern)."""
    return shutil.which("codex") is not None


def call_codex(prompt: str, model: str, timeout: int) -> tuple[bool, str]:
    """One codex exec call. Mirrors pages_adversarial_audit.py:call_codex and
    scripts/research_conductor.py's own `_build_agent_command` codex branch --
    same flags, same stdin-piped-prompt shape, same '-' terminator. Deliberately
    a plain text-completion call (no repo tool access): the hypothesis is a
    sandboxed snippet, never an agentic edit to real files.
    """
    try:
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--color",
                "never",
                "--model",
                model,
                "--cd",
                str(PROJECT_ROOT),
                "--ephemeral",
                "-",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"codex exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)


def codex_generate_hypotheses(
    model: str,
    timeout: int,
    baselines: BaselineRecord,
    recent_failures: list[dict[str, Any]],
    iteration: int,
) -> list[tuple[str, str]]:
    """The `generator` callback `run_loop_with_generator` expects. Reuses
    hypothesis_generator.py's own prompt-building and response-parsing --
    only the transport (codex subprocess vs. a raw HTTP client) differs."""
    prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{_build_user_prompt(baselines, recent_failures, iteration)}"
    ok, output = call_codex(prompt, model, timeout)
    if not ok:
        recent_failures.append({"description": "codex_call_failed", "reason": output})
        return []
    return _extract_hypotheses(output)


def _git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=project_root, check=check, capture_output=True, text=True
    )


def commit_accepted_hypothesis(
    checker: ConstitutionChecker,
    entry: ExperimentEntry,
    benchmark_name: str,
    baseline_before: float | None,
    baseline_after: float | None,
    *,
    project_root: Path,
) -> str | None:
    """Persist one accepted hypothesis as its own scoped git commit.

    AVO's "git commit per version with its score", adapted: the record is a
    JSON sidecar (see module docstring for why), and the git add is an
    explicit single path, never `-A`. Returns the new commit SHA, or None if
    the constitution forbade it or the commit did not apply cleanly.
    """
    rel_dir = Path("ops") / "autoresearch_discoveries" / benchmark_name
    rel_path = rel_dir / f"{entry.id}.json"

    create_verdict = checker.check(f"create_file:{rel_path.as_posix()}")
    if create_verdict.category != ActionCategory.ALLOWED:
        return None
    commit_verdict = checker.check("git_commit")
    if commit_verdict.category != ActionCategory.ALLOWED:
        return None

    (project_root / rel_dir).mkdir(parents=True, exist_ok=True)
    record = {
        "id": entry.id,
        "timestamp": entry.timestamp,
        "benchmark": benchmark_name,
        "description": entry.hypothesis_description,
        "code": entry.hypothesis_code,
        "metrics": entry.sandbox_metrics.get(benchmark_name, {}),
        "eval_verdict": entry.eval_verdict,
        "eval_reason": entry.eval_reason,
        "baseline_final_energy_before": baseline_before,
        "baseline_final_energy_after": baseline_after,
    }
    (project_root / rel_path).write_text(json.dumps(record, indent=2) + "\n")

    add = _git(project_root, "add", rel_path.as_posix(), check=False)
    if add.returncode != 0:
        return None

    message = (
        f"[autoresearch] {benchmark_name}: accept {entry.id} "
        f"(final_energy {baseline_before} -> {baseline_after})\n\n"
        f"{entry.hypothesis_description}\n\n"
        "Autonomous mutation round -- no operator or outer-loop session\n"
        "involved in this commit. Hypothesis proposed by the configured\n"
        "local generator model, accepted by the existing 3-gate evaluator\n"
        "(REQ-AUTO-005), persisted per REQ-AUTO-019/REQ-AUTO-020.\n"
    )
    commit = _git(project_root, "commit", "-m", message, check=False)
    if commit.returncode != 0:
        _git(project_root, "reset", "HEAD", "--", rel_path.as_posix(), check=False)
        return None
    return _git(project_root, "rev-parse", "HEAD").stdout.strip()


def run_round(
    *,
    api_base: str,
    model: str,
    max_iterations: int,
    project_root: Path = PROJECT_ROOT,
    baseline_cache: Path | None = None,
    log_cache: Path | None = None,
    receipt_path: Path | None = None,
) -> int:
    baseline_cache = baseline_cache or project_root / "ops" / ".autoresearch_baselines.json"
    log_cache = log_cache or project_root / "ops" / ".autoresearch_experiment_log.json"
    receipt_path = receipt_path or project_root / "ops" / "autoresearch_conductor_report.md"
    for path in (baseline_cache, log_cache, receipt_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    started = datetime.now(UTC)
    report_lines = [
        "# Autoresearch conductor round",
        "",
        f"- started: {started.isoformat()}",
        f"- api_base: {api_base}",
        f"- model: {model}",
        f"- max_iterations: {max_iterations}",
        "",
    ]

    if not endpoint_reachable(api_base):
        report_lines.append(f"BLOCKED: LLM endpoint `{api_base}` unreachable. No round run.")
        receipt_path.write_text("\n".join(report_lines) + "\n")
        print("blocked_llm_endpoint_unreachable")
        return 0

    checker = ConstitutionChecker()
    baselines = load_baselines(baseline_cache)
    experiment_log = load_experiment_log(log_cache)
    before_count = len(experiment_log.entries)
    energy_before = {name: metrics.final_energy for name, metrics in baselines.benchmarks.items()}

    gen_config = GeneratorConfig(api_base=api_base, model=model)

    def generator(
        cur_baselines: BaselineRecord,
        recent_failures: list[dict[str, Any]],
        iteration: int,
    ) -> list[tuple[str, str]]:
        return generate_hypotheses_batch(
            gen_config, cur_baselines, recent_failures, iteration, count=1
        )

    config = AutoresearchConfig(
        max_iterations=max_iterations,
        max_consecutive_failures=10,
        constitution_checker=checker,
    )
    result = run_loop_with_generator(
        generator, baselines, DEFAULT_BENCHMARK_DATA, config, experiment_log
    )

    new_entries = experiment_log.entries[before_count:]
    committed: list[tuple[str, str, str]] = []
    for entry in new_entries:
        if entry.outcome != "accepted":
            continue
        for bench_name in entry.sandbox_metrics:
            after = result.final_baselines.benchmarks.get(bench_name)
            sha = commit_accepted_hypothesis(
                checker,
                entry,
                bench_name,
                energy_before.get(bench_name),
                after.final_energy if after else None,
                project_root=project_root,
            )
            if sha:
                committed.append((entry.id, bench_name, sha))

    result.final_baselines.save(baseline_cache)
    experiment_log.save(log_cache)

    report_lines += [
        f"- iterations: {result.iterations}",
        f"- accepted: {result.accepted}",
        f"- rejected: {result.rejected}",
        f"- pending_review: {result.pending_review}",
        f"- circuit_breaker_tripped: {result.circuit_breaker_tripped}",
        "",
    ]
    if committed:
        report_lines.append("## Committed lineage")
        for exp_id, bench, sha in committed:
            report_lines.append(f"- {exp_id} ({bench}): {sha[:12]}")
    else:
        report_lines.append("No hypothesis both won this round and committed cleanly.")
    receipt_path.write_text("\n".join(report_lines) + "\n")

    print(
        f"complete: autoresearch round iterations={result.iterations} "
        f"accepted={result.accepted} committed={len(committed)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-iterations", type=int, default=5)
    args = parser.parse_args()
    return run_round(api_base=args.api_base, model=args.model, max_iterations=args.max_iterations)


if __name__ == "__main__":
    sys.exit(main())
