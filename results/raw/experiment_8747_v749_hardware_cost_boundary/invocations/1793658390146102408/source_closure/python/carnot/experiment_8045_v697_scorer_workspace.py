"""REQ-REPORT-8045: qualify the scorer's test venue without loading weights.

The numerical controls deliberately use tiny fake native contexts. They expose
buffer ownership and target alignment mistakes, but cannot prove repeatability
of a live model. Exp8033 remains disqualified with its original bytes intact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot import experiment_8033_v696_scoring_isolation as old
from carnot.inference import scoring_isolation_8033 as scorer
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.experiment_8023_v695_likelihood_calibration import terminal_readers

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8045_v697_scorer_workspace"
TASK = "exp8045-scorer-workspace"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_scorer_workspace_8045.py"
FAILURE = "results/raw/experiment_8033_v696_scoring_isolation/validation_logs/01_focused_pytest.log"
HISTORY = "results/experiment_8033_v696_scoring_isolation.json"
SIDECAR = "results/raw/experiment_8033_v696_scoring_isolation/terminal_validation.json"
INPUTS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    *old.OWNED[:-1],
    old.TEST,
    FAILURE,
    HISTORY,
    SIDECAR,
    "python/carnot/reporting/experiment_7303_validation_scope.py",
]
METHODS = dict(
    seed=69745,
    condition="fresh_full",
    sources=2,
    calls=4,
    target_tokens=2,
    duplicate_tolerance=1e-6,
    normalization_tolerance=1e-10,
    child_timeout_s=180,
    fixture_timeout_s=30,
    health_timeout_s=60,
    condition_for_exp8047="fresh_full",
    condition_for_exp8049="fresh_full",
    tokenizer="embedded_GGUF",
    generation=False,
    selection="fixed_before_measurement",
)


class Context:
    """Keep an independent position counter so leaked context state is visible."""

    def __init__(self, **kwargs: Any) -> None:
        self.ctx = id(self)
        self.memory = self.ctx
        self.position = -1
        self.closed = False

    def kv_cache_seq_rm(self, seq: int, start: int, stop: int) -> bool:
        """Mirror removal of earlier positions; missed resets leave a detectable offset."""
        self.position = start - 1
        return True

    def decode(self, batch: Any) -> None:
        """Advance exactly the submitted tokens, exposing shifted decode positions."""
        self.position = batch.n_past + batch.n_tokens - 1

    def close(self) -> None:
        """Mark lifetime completion so fixture success requires normal cleanup."""
        self.closed = True


class Model:
    """A tiny owned logit array supplies known probabilities without model calls."""

    def __init__(self) -> None:
        self._ctx = Context()
        self._model = object()
        self.context_params = SimpleNamespace(n_ubatch=256)
        self.n_batch, self.n_tokens, self.reset_count = 256, 99, 0
        self.scores = np.zeros((4, 3))

    def reset(self) -> None:
        """Erase the deliberately dirty Python token count before every score."""
        self.n_tokens = 0
        self.reset_count += 1

    def eval(self, tokens: list[int]) -> None:
        """Run the controller's normal removal and decode hooks with known logits."""
        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        batch = SimpleNamespace(n_past=self.n_tokens, n_tokens=len(tokens))
        batch.batch = batch
        self._ctx.decode(batch)
        self.n_tokens += len(tokens)
        self.scores[:] = [0.0, math.log(2.0), math.log(3.0)]


def fixture_rows() -> list[Json]:
    """Exercise the shipped controller and poison its buffer after scalar capture."""
    from llama_cpp import _internals, llama_cpp as native

    model = Model()
    contexts: list[Context] = []

    def factory(**kwargs: Any) -> Context:
        context = Context(**kwargs)
        contexts.append(context)
        return context

    rows = []
    with (
        patch.object(_internals, "LlamaContext", factory),
        patch.object(native, "llama_memory_seq_pos_min", return_value=-1),
        patch.object(
            native, "llama_memory_seq_pos_max", side_effect=lambda *a: model._ctx.position
        ),
    ):
        controller = scorer.NativeController(SimpleNamespace(model=model), time.monotonic() + 30)
        for i in range(4):
            model.n_tokens = 99
            result = controller.score(dict(tokens=[0, 0, 1, 2], response_start=2), "fresh_full")
            saved = list(result["target_logprobs"])
            model.scores[:] = 100
            rows.append(
                dict(
                    result,
                    source_id=f"fixture-{i % 2}",
                    seed=69745,
                    condition="fresh_full",
                    tokens=[0, 0, 1, 2],
                    target_tokens=[1, 2],
                    response_start=2,
                    reset_observed=model.reset_count == i + 1,
                    copied_scores_survived=result["target_logprobs"] == saved,
                    context_closed=contexts[-1].closed,
                    numerator=math.fsum(-p for p in saved),
                    denominator=2,
                    independent_count=0,
                )
            )
    return rows


def reduce_rows(rows: list[Json]) -> Json:
    """Reduce every token against a known distribution without importing old drift."""
    expected = [-math.log(3.0), -math.log(2.0)]
    checks = [
        dict(
            source_id=r["source_id"],
            condition="fresh_full",
            numerator=r["numerator"],
            denominator=r["denominator"],
            independent_count=0,
            passed=r["tokens"] == [0, 0, 1, 2]
            and r["target_tokens"] == [1, 2]
            and r["response_start"] == 2
            and r["conditional_logit_positions"] == [1, 2]
            and r["reset_observed"]
            and r["copied_scores_survived"]
            and r["context_closed"]
            and r["normalization_max_error"] <= 1e-10
            and all(
                abs(a - b) <= 1e-12 for a, b in zip(r["target_logprobs"], expected, strict=True)
            ),
        )
        for r in rows
    ]
    return dict(
        rows=checks,
        passed=len(rows) == 4
        and all(r["passed"] for r in checks)
        and len({r["context_identity"] for r in rows}) == 4,
    )


def preconditions(root: Path) -> list[Json]:
    """Name missing local operands before any measurement, rather than fill zeros."""
    paths = [root / p for p in INPUTS]
    paths += [ROOT / ".venv/bin" / p for p in ("python", "pytest", "coverage", "ruff", "mypy")]
    checks = [
        old.operand(
            p,
            "resource_exists",
            True,
            True if p.is_file() else "missing_resource",
            "exp8033" if "8033" in str(p) else "exp8045_input",
        )
        for p in paths
    ]
    history = root / HISTORY
    if history.is_file():
        value = json.loads(history.read_text())
        checks += [
            old.operand(history, k, v, value.get(k, "missing_field_contract_error"), "exp8033")
            for k, v in dict(verdict_class="disqualified", scoring_isolation_ready_score=0).items()
        ]
        sidecar = json.loads((root / SIDECAR).read_text()) if (root / SIDECAR).is_file() else {}
        checks.append(
            old.operand(
                root / SIDECAR,
                "publication.primary_sha256",
                reference(history)["sha256"],
                sidecar.get("publication", {}).get(
                    "primary_sha256", "missing_field_contract_error"
                ),
                "exp8033",
            )
        )
    return checks


def reproduce(scratch: Path) -> Json:
    """Use real pytest to retain the original missing-parent failure and its repair."""
    scratch.mkdir(parents=True, exist_ok=True)
    fixture = scratch / "test_private_workspace.py"
    fixture.write_text("def test_workspace(tmp_path):\n    assert tmp_path.is_dir()\n")
    artifacts = scratch / "artifacts"
    artifacts.mkdir()
    (scratch / "conftest.py").write_text(
        f"from carnot.testing.child_results_guard import install\ninstall({str(artifacts)!r})\n"
    )
    specs = [
        CommandSpec(
            "missing_parent",
            (
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                str(fixture),
                f"--basetemp={scratch / 'pytest/focused'}",
            ),
            "control",
            30,
        )
    ]
    before = run_commands(ROOT, specs, log_dir=scratch / "before", heartbeat_s=10)[0]
    absent = not (scratch / "pytest").exists()
    (scratch / "pytest").mkdir(parents=True, exist_ok=True)
    after = run_commands(
        ROOT, [replace(specs[0], name="repaired_parent")], log_dir=scratch / "after", heartbeat_s=10
    )[0]
    return dict(
        before=dict(
            before,
            expected_exit_code=1,
            actual_exit_code=before["exit_code"],
            missing_parent_detected=absent and "FileNotFoundError" in before["output_tail"],
        ),
        after=dict(
            after,
            expected_exit_code=0,
            actual_exit_code=after["exit_code"],
            parent_exists_before_launch=True,
        ),
    )


def commands(scratch: Path) -> list[CommandSpec]:
    """Freeze only task-owned acceptance; a full-suite timeout is separate health."""
    old.commands(scratch)
    py = str(ROOT / ".venv/bin/python")
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in [MODULE, CLI, old.OWNED[0]])
    )
    tests = [TEST, old.TEST]
    pytest = (str(ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q")
    specs = [
        CommandSpec(
            "focused_pytest",
            (*pytest, f"--basetemp={scratch / 'pytest/focused'}", *tests),
            "owned",
            180,
        ),
        CommandSpec(
            "consumer_contracts",
            (
                *pytest,
                f"--basetemp={scratch / 'pytest/consumer'}",
                "tests/python/test_likelihood_protocol_8022.py",
                "tests/python/test_primary_publication_7928.py",
            ),
            "owned",
            180,
        ),
        CommandSpec(
            "combined_cli_coverage",
            (
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *pytest[1:],
                f"--basetemp={scratch / 'pytest/coverage'}",
                *tests,
            ),
            "owned",
            180,
        ),
        CommandSpec(
            "combine", (py, "-m", "coverage", "combine", "--rcfile=" + str(config)), "owned", 30
        ),
        CommandSpec(
            "coverage_json",
            (
                py,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(scratch / "coverage.json"),
            ),
            "owned",
            30,
        ),
    ]
    files = [MODULE, CLI, TEST, old.OWNED[0]]
    specs += [
        CommandSpec("ruff_check", (str(ROOT / ".venv/bin/ruff"), "check", *files), "owned", 30),
        CommandSpec(
            "ruff_format", (str(ROOT / ".venv/bin/ruff"), "format", "--check", *files), "owned", 30
        ),
        CommandSpec(
            "strict_mypy",
            (
                str(ROOT / ".venv/bin/mypy"),
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                MODULE,
                old.OWNED[0],
            ),
            "owned",
            60,
        ),
        CommandSpec(
            "spec_coverage", (py, "-u", "scripts/check_spec_coverage.py", *tests), "owned", 30
        ),
        CommandSpec(
            "repository_health",
            (
                "timeout",
                "--kill-after=5s",
                "60s",
                str(ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
            ),
            "repository_health",
            65,
        ),
    ]
    return specs


def validate(scratch: Path, raw: Path) -> Json:
    """Keep every child exit and log, and count only statements added by this task."""
    specs = commands(scratch)
    atomic_json(raw / "validation_commands.json", dict(commands=[asdict(x) for x in specs]))
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=raw / "validation_logs",
        heartbeat_s=20,
        extra_env=dict(
            CARNOT_8045_COVERAGE_CONFIG=str(scratch / "coverage.ini"), JAX_PLATFORMS="cpu"
        ),
    )
    report = (
        json.loads((scratch / "coverage.json").read_text())
        if (scratch / "coverage.json").is_file()
        else dict(files={})
    )
    selected = [v for k, v in report["files"].items() if k in {MODULE, CLI}]
    repair_line = next(
        i
        for i, line in enumerate((ROOT / old.OWNED[0]).read_text().splitlines(), 1)
        if '(scratch / "pytest").mkdir' in line
    )
    original = report["files"].get(old.OWNED[0], {})
    executed = repair_line in original.get("executed_lines", [])
    total = sum(v["summary"]["num_statements"] for v in selected) + 1
    covered = sum(v["summary"]["covered_lines"] for v in selected) + int(executed)
    coverage = dict(
        num_statements=total,
        covered_lines=covered,
        missing_lines=total - covered,
        percent_covered=100 * covered / total,
        repaired_boundary_executed=executed,
        files={k: v for k, v in report["files"].items() if k in {MODULE, CLI}},
        required_percent=100,
        original_module_changed_lines=[repair_line],
    )
    receipts = [dict(r, expected_exit_code=0, actual_exit_code=r["exit_code"]) for r in receipts]
    return dict(receipts=receipts, coverage=coverage)


def build(plan: Json, work: Json, raw: Path, validation: Json) -> Json:
    """Readiness binds code and venue qualification, with no scientific sample credit."""
    rows = work["rows"]
    reduced = reduce_rows(rows)
    owned = [r for r in validation["receipts"] if r["scope"] == "owned"]
    coverage = validation["coverage"]
    workspace = work.get("workspace", {})
    gates = dict(
        inputs=all(r["passed"] for r in plan["checks"]),
        fixture=reduced["passed"],
        workspace=workspace.get("before", {}).get("missing_parent_detected", False)
        and workspace.get("before", {}).get("actual_exit_code") == 1
        and workspace.get("after", {}).get("actual_exit_code") == 0,
        owned_checks=bool(plan["acceptance_manifest"])
        and [r["name"] for r in owned] == plan["acceptance_manifest"]
        and all(r["passed"] for r in owned),
        coverage=coverage.get("percent_covered") == 100 and coverage.get("missing_lines") == 0,
    )
    failed = [r for r in plan["checks"] if not r["passed"]]
    verdict = "blocked" if failed else "circular_positive" if all(gates.values()) else "null"
    sizes = dict(
        intended_count=4,
        eligible_count=4 if gates["inputs"] else 0,
        completed_count=len(rows),
        excluded_count=0,
        failed_count=0,
        censored_count=4 - len(rows),
        independent_count=0,
    )
    value = dict(
        experiment_id=8045,
        task_id=TASK,
        milestone="2026.10.697",
        schema="carnot.v697.scorer_workspace.v1",
        run_date="20261003",
        claim_scope="This invocation qualifies CPU scorer fixtures and validation workspaces only; live repeatability and deployment correctness remain unproven.",
        honest_verdict="complete_blocked_" + Path(failed[0]["path"]).stem.replace(".", "_")
        if failed
        else "complete_" + verdict + "_scorer_workspace",
        verdict_class=verdict,
        gate_check_summary=plan["checks"],
        rows=reduced["rows"],
        sample_size_budget=dict(
            **sizes,
            forward_call_budget=4,
            source_count=2,
            seed_count=1,
            unit="oracle fixture calls; no independent scientific observations",
        ),
        **sizes,
        random_seed=69745,
        reproducibility_checksum=canonical_hash(dict(plan=plan, work=work, validation=validation)),
        cited_upstream_artifacts=plan["inputs"],
        code_config_hashes=plan["code"],
        raw_shard_hashes=[
            reference(raw / p)
            for p in (
                "plan.json",
                "work.json",
                "fixture.json",
                "validation.json",
                "validation_commands.json",
            )
        ],
        checkpoint_references=[reference(raw / "fixture.json")],
        acceptance_gate_results=gates,
        verifier_is_oracle=True,
        genuine_headroom=None,
        positive_control_results=dict(
            scorer.oracle(), scope="known logit distribution; no live evidence"
        ),
        generalized_learning_benefit_score=0,
        validation_receipts=owned,
        repository_health=[r for r in validation["receipts"] if r["scope"] == "repository_health"],
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=False,
        preconditions_checked=plan["checks"],
        inference_substrate="artifact_qa_lint_tests",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_model_invocation_count=0,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        scorer_fixture_ready_score=int(all(gates.values())),
        scorer_code_hashes=plan["code"],
        fixed_condition="fresh_full",
        methods=METHODS,
        original_failure_log=plan["original_failure_log"],
        repaired_workspace_receipts=workspace,
        token_alignment_control_rows=rows,
        historical_output_qualification="Exp8033 remains disqualified; its measured drift is not requalified.",
        substrate_declaration=dict(
            substrate="verifier_scoring",
            reduction="aggregation_from_upstream_artifacts",
            mode="no_model_load",
            MODEL_SPECS=[],
            pretrained_model_calls=0,
        ),
        methodology_note="Exact probabilities come from a small known distribution and exposed oracle controls. Fixture readiness is not live repeatability.",
    )
    value["field_principles"] = {
        k: f"Record {k} for this invocation; bind claims to original bytes and actual exits."
        for k in value
    }
    for field in (
        "scorer_fixture_ready_score",
        "verifier_is_oracle",
        "positive_control_results",
        "generalized_learning_benefit_score",
    ):
        value["field_principles"][field] = (
            "Certify exposed code/fixture qualification only; oracle controls cannot qualify live repeatability, historical output or deployment correctness."
        )
    for field in ("rows", "sample_size_budget", *sizes, "token_alignment_control_rows"):
        value["field_principles"][field] = (
            "Retain each source, seed, condition, numerator and denominator; repeated calls and seeds add no independent scientific observations."
        )
    for field in (
        "code_config_hashes",
        "scorer_code_hashes",
        "raw_shard_hashes",
        "checkpoint_references",
        "cited_upstream_artifacts",
        "reproducibility_checksum",
    ):
        value["field_principles"][field] = (
            "Authenticate original bytes and frozen code/configuration; reject missing or corrupted shards rather than substitute mutable summaries."
        )
    value["field_principles"]["gate_check_summary"] = (
        "Name each exact upstream operand, expected and observed value; absence is a contract failure, never a measured zero."
    )
    value["field_principles"]["original_failure_log"] = (
        "Preserve the original missing-parent failure without retroactively qualifying disqualified Exp8033 model measurements."
    )
    value["field_principles"]["substrate_declaration"] = (
        "Separate upstream reduction from CPU verifier scoring; no model load, no generation and zero pretrained calls."
    )
    return value


def replay(value: Json) -> None:
    """Reopen primitive rows and every cited byte before recomputing the whole claim."""
    for ref in (
        value["raw_shard_hashes"]
        + value["checkpoint_references"]
        + value["code_config_hashes"]
        + value["cited_upstream_artifacts"]
    ):
        checked(ref)
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    plan, work, validation = [
        json.loads((raw / p).read_text()) for p in ("plan.json", "work.json", "validation.json")
    ]
    if json.loads((raw / "fixture.json").read_text())["rows"] != work["rows"]:
        raise ValueError("fixture_row_drift")
    for receipt in (
        value["validation_receipts"] + value["repository_health"] + list(work["workspace"].values())
    ):
        checked(dict(path=receipt["log_path"], sha256=receipt["log_sha256"]))
    if build(plan, work, raw, validation) != value:
        raise ValueError("cold_reduction_drift")


def terminal(path: Path) -> Json:
    """The existing validators inspect the same bytes as independent cold reduction."""
    replay(json.loads(path.read_text()))
    return terminal_readers(path)


def main(argv: list[str] | None = None) -> int:
    """Freeze and measure once, then publish through the shipped primary helper."""
    started = time.monotonic()
    scorer.progress("8045_start_preconditions", started)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--shift-position", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            return 0
        if args.fixture_output:
            rows = fixture_rows()
            if args.shift_position:
                rows[0]["conditional_logit_positions"][0] += 1
            atomic_json(args.fixture_output, dict(rows=rows))
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        if (raw / "work.json").exists():
            raise ValueError("original_work_preserved_use_cold_replay")
        raw.mkdir(parents=True, exist_ok=True)
        checks = preconditions(args.root)
        sources = [args.root / p for p in INPUTS if (args.root / p).is_file()]
        durable = []
        for source in sources:
            target = raw / "inputs" / source.relative_to(args.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            durable.append(reference(target))
        code = [
            reference(ROOT / p)
            for p in [
                MODULE,
                CLI,
                TEST,
                *old.OWNED[:-1],
                old.TEST,
                "python/carnot/inference/fixed_answer_likelihood_8022.py",
                "python/carnot/inference/likelihood_runtime_8022.py",
            ]
        ]
        phases = [dict(phase="preconditions", start_s=0.0, end_s=time.monotonic() - started)]
        with TemporaryDirectory(prefix="carnot-8045-") as directory:
            scratch = Path(directory)
            specs = commands(scratch) if all(r["passed"] for r in checks) else []
            atomic_json(raw / "validation_commands.json", dict(commands=[asdict(s) for s in specs]))
            plan = dict(
                checks=checks,
                inputs=durable,
                code=code,
                methods=METHODS,
                acceptance_manifest=[s.name for s in specs if s.scope == "owned"],
                original_failure_log=next(
                    (r for r in durable if r["path"].endswith(FAILURE)), None
                ),
            )
            atomic_json(raw / "plan.json", plan)
            rows, workspace, validation = [], {}, dict(receipts=[], coverage={})
            if all(r["passed"] for r in checks):
                began = time.monotonic() - started
                scorer.progress("8045_before_workspace_control", started)
                workspace = reproduce(scratch / "reproduction")
                shutil.copytree(scratch / "reproduction", raw / "workspace", dirs_exist_ok=True)
                workspace = old.relocate(workspace, scratch / "reproduction", raw / "workspace")
                scorer.progress("8045_after_workspace_control", started)
                child = run_commands(
                    scratch,
                    [
                        CommandSpec(
                            "cpu_fixture_cli",
                            (
                                "env",
                                "-u",
                                "PYTHONPATH",
                                str(ROOT / ".venv/bin/python"),
                                "-u",
                                str(ROOT / CLI),
                                "--fixture-output",
                                str(scratch / "fixture.json"),
                            ),
                            "fixture",
                            30,
                        )
                    ],
                    log_dir=raw / "fixture_logs",
                    heartbeat_s=10,
                )[0]
                rows = (
                    json.loads((scratch / "fixture.json").read_text())["rows"]
                    if child["passed"]
                    else []
                )
                workspace["fixture_cli_exit"] = dict(
                    child, expected_exit_code=0, actual_exit_code=child["exit_code"]
                )
                phases.append(
                    dict(phase="cpu_fixture", start_s=began, end_s=time.monotonic() - started)
                )
                scorer.progress("8045_before_validation", started)
                began = time.monotonic() - started
                validation = validate(scratch, raw)
                phases.append(
                    dict(phase="validation", start_s=began, end_s=time.monotonic() - started)
                )
                scorer.progress("8045_after_validation", started)
            atomic_json(raw / "fixture.json", dict(rows=rows))
            work = dict(
                rows=rows,
                workspace=workspace,
                duration_s=time.monotonic() - started,
                phase_spans=phases,
            )
            atomic_json(raw / "work.json", work)
            atomic_json(raw / "validation.json", validation)
            value = build(plan, work, raw, validation)
            candidate = raw / "candidate.json"
            atomic_json(candidate, value)
            cold = run_commands(
                ROOT,
                [
                    CommandSpec(
                        "cold_reduction",
                        (
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            str(ROOT / CLI),
                            "--cold-replay",
                            str(candidate),
                        ),
                        "terminal",
                        60,
                    )
                ],
                log_dir=raw / "cold_logs",
            )[0]
            if not cold["passed"]:
                raise ValueError("cold_reduction_failed")
            publication = publish_primary(output, value, terminal)
            published = terminal(output)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="scorer_fixture_ready_score",
                expected=value["scorer_fixture_ready_score"],
            )
            bound = read_bound_sidecar(output, Path(publication["sidecar_path"]))
            atomic_json(
                raw / "terminal_validation.json",
                dict(
                    publication=publication,
                    cold=cold,
                    published=published,
                    readers=readers,
                    primary_sha256=publication["primary_sha256"],
                    bound_report_passed=bound["report"]["passed"],
                ),
            )
            if not published["passed"] or not readers["passed"]:
                raise ValueError("published_validation_failed")
        scorer.progress("8045_complete", started, len(rows), 0)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TimeoutError) as error:
        print(f"[exp8045] rejected={type(error).__name__}:{error}", flush=True)
        return 1
