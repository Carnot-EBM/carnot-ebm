"""REQ-REPORT-8405: current qualification preserves finite arithmetic and old failures.

Only authority and evidence custody change. The shipped binding, numerical
panel and terminal consumers still decide what this invocation can claim.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator
from unittest.mock import patch

import coverage

from carnot.reporting import historical_consumer_roots_8402 as current
from carnot.reporting import native_direct_8379 as base
from carnot.reporting import native_evidence_8391 as old
from carnot.reporting import native_direct_runner_8379 as runner
from carnot.reporting import v723_contract_methods as historical
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = (
    "experiment_8405_v724_native_parity_qualification",
    "exp8405-native-parity-qualification",
)
CLI, TEST = (
    f"scripts/experiments/{NAME}.py",
    "tests/python/test_native_parity_qualification_8405.py",
)
OWNED = ["python/carnot/reporting/native_parity_8405.py", CLI]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:793c5be8b25032f4558ed36e82f2c1e70bad7edc412a5cbc045268fcf5551eb6"
ORIGINAL = "results/experiment_8391_v723_native_evidence_qualification.json"
ORIGINAL_PIN = "sha256:80f9bd09515fe6492ef184b76e995268591c7ab192f68393988f0f326e943a81"
PRESERVED = [*old.PRESERVED, ORIGINAL, historical.PROTOCOL]
REPAIRED = "python/carnot/reporting/native_evidence_8391.py"
reference = old.reference
OLD_BIND, OLD_BUILD, OLD_COVERAGE = old.bind, old.build, old.coverage_evidence
OLD_AUDIT = runner.audit


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts keep a stalled owned process visible to its supervisor."""
    print(f"[exp8405] phase={phase} completed={completed} pending={pending}", flush=True)


def bind(root: Path, raw: Path) -> tuple[Json | None, list[Json], list[Json]]:
    """Exact current task authority cannot borrow an earlier protocol's activation."""
    with (
        patch.multiple(old, TASK=TASK, TASK_PIN=TASK_PIN, reference=reference),
        patch.object(historical, "authority", current.authority),
    ):
        protocol, refs, gates = OLD_BIND(root, raw)
    for name, pin in [
        (current.PROTOCOL, current.PROTOCOL_PIN),
        (current.METHODS, current.METHODS_PIN),
        (ORIGINAL, ORIGINAL_PIN),
    ]:
        source = root / name
        observed = reference(source)["sha256"] if source.is_file() else None
        if observed != pin:
            gates.append(current.failure(source, "v724_input_hash", pin, observed))
        else:
            saved = raw / "inputs" / (pin[7:] + source.suffix)
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(source.read_bytes())
            refs.append(dict(reference(source), snapshot_path=str(saved)))
    return (None if gates else protocol), refs, gates


@contextmanager
def bindings() -> Iterator[None]:
    """Process-local adapters preserve historical modules and their date contracts."""
    with (
        patch.multiple(
            old,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            bind=bind,
            build=build,
            progress=progress,
        ),
        patch.object(historical, "authority", current.authority),
        old.bindings(),
    ):
        yield


def coefficient_reduction(rows: list[Json]) -> Json:
    """Paired operands prove a zero error; a changed operand must break that proof."""
    pairs = [r for r in rows if r.get("kind") == "numeric_update"]
    errors = [
        max(
            abs(a - b)
            for a, b in zip(r["python_coefficients"], r["native_coefficients"], strict=True)
        )
        for r in pairs
    ]
    matched = bool(pairs) and all(
        v == r["coefficient_error"] for v, r in zip(errors, pairs, strict=True)
    )
    negative = False
    if pairs:
        changed = list(pairs[0]["native_coefficients"])
        changed[0] += 0.125
        negative = (
            max(abs(a - b) for a, b in zip(pairs[0]["python_coefficients"], changed, strict=True))
            != pairs[0]["coefficient_error"]
        )
    return dict(
        recomputed=matched,
        deliberate_error_rejected=negative,
        paired_update_count=len(pairs),
        max_error=max(errors, default=None),
        operand_sha256=canonical_hash(pairs),
    )


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Reuse actual native calls and sealed historical replay without inventing results."""
    with bindings():
        work = dict(old.measure(root, raw, private))
    progress("independent_coefficient_reduction_before")
    work["coefficient_reduction"] = coefficient_reduction(work["rows"])
    work["code_config_hashes"].extend(
        reference(ROOT / p)
        for p in [
            REPAIRED,
            "python/carnot/reporting/historical_consumer_roots_8402.py",
            "python/carnot/reporting/v718_replay_history.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "python/carnot/reporting/primary_publication.py",
        ]
    )
    work["original_qualification"] = dict(reference(ROOT / ORIGINAL), readiness_promoted=False)
    work["code_config_hashes"].extend(
        reference(Path(module.__file__))
        for module in [
            base,
            base.k,
            runner,
            base.k.optimizer,
            base.k.kernel,
            base.k.evaluator,
        ]
    )
    atomic_json(raw / "measurement.json", work)
    progress("independent_coefficient_reduction_after", len(work["rows"]), 0)
    return work


def shard_identity(path: Path) -> Json:
    """Reading Coverage.py's database checks schema and source identities before combine."""
    data = coverage.CoverageData(basename=str(path))
    data.read()
    files = sorted(data.measured_files())
    return dict(
        reference(path),
        measured_files=files,
        contexts=sorted(data.measured_contexts()),
        coverage_version=coverage.__version__,
        database_format="coverage_sqlite",
        line_identity=canonical_hash({name: sorted(data.lines(name) or []) for name in files}),
    )


def retain_shards(private: Path) -> int:
    """Preserve real database identities before report commands remove original shards."""
    result = old.retain_shards(private)
    saved = private / "coverage_shards"
    identities = [shard_identity(p) for p in sorted(saved.glob(".coverage.*"))]
    atomic_json(saved / "identities.json", dict(files=identities, invocation=str(private)))
    return result


def coverage_evidence(private: Path) -> Json:
    """A combined report must agree with database lines and original shard identities."""
    with patch.multiple(old, ROOT=ROOT, OWNED=OWNED):
        result = OLD_COVERAGE(private)
    identities = private / "coverage_shards/identities.json"
    if not identities.is_file():
        return dict(result, passed=False, database_identities_verified=False)
    value = json.loads(identities.read_bytes())
    verified = (
        value["invocation"] == str(private)
        and sum(bool(ref["measured_files"]) for ref in value["files"]) >= 2
    )
    for ref in value["files"]:
        verified = verified and shard_identity(Path(ref["path"])) == ref
    return dict(
        result,
        passed=result["passed"] and verified,
        database_identities_verified=verified,
        database_identities=value["files"],
    )


def build(work: Json, receipts: list[Json]) -> Json:
    """Typing and finite parity qualification never upgrade the closed learning nulls."""
    with patch.object(old, "coverage_evidence", coverage_evidence):
        value = dict(OLD_BUILD(work, receipts))
    proof = coefficient_reduction(work["rows"])
    if work["input_ready"] and not (proof["recomputed"] and proof["deliberate_error_rejected"]):
        value.update(
            native_parity_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=True,
            verdict_class="disqualified",
        )
    value.update(
        experiment_id=8405,
        task_id=TASK,
        milestone="2026.10.724",
        run_date="20261011",
        honest_verdict="complete_" + value["verdict_class"] + "_native_parity_qualification",
        coefficient_reduction=proof,
        strict_mypy_receipt=next((r for r in receipts if r.get("name") == "strict_mypy"), None),
        original_qualification=work["original_qualification"],
        binding_ownership="PyO3 owns copied Vec<f64> inputs and returns Python lists; allocator overhead is unmeasured",
        future_dependencies=["exp8407-python-transaction-cost", "exp8408-native-transaction-cost"],
        repository_health=dict(
            status="unknown_current_global_health",
            prior_observations=work["repository_health"],
            rerun=False,
        ),
        model_invocation_counts=dict(
            current_llm_calls=0,
            current_model_loads=0,
            current_generation_calls=0,
            numeric_heads=int(bool(work["rows"])),
        ),
    )
    value["acceptance_gates"]["independent_coefficient_reduction"] = bool(proof["recomputed"])
    value["field_principles"].update(
        {
            k: "Bind actual task, source and paired finite operands; unchanged consumers retain their decisions and no benefit follows."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh source-bound native execution defeats self-consistent aggregate rehashing."""
    with bindings():
        return bool(old.replay(path))


def audit(candidate: Path, raw: Path, ignored: Json) -> Json:
    """Pass independently reduced operands to the unchanged typed finding consumer."""
    value = json.loads(candidate.read_bytes())
    return dict(OLD_AUDIT(candidate, raw, coefficient_reduction(value["rows"])))


def plan(private: Path) -> list[Json]:
    """Freeze the current owned scope before opening measured outputs or coverage."""
    with bindings():
        commands = old.plan(private)
    config = private / "coverage.ini"
    text = config.read_text().replace("[report]", "    " + str(ROOT / REPAIRED) + "\n[report]")
    config.write_text(text)
    for command in commands:
        if command["name"] == "strict_mypy":
            command["argv"].append(REPAIRED)
        if command["name"] in {"ruff_check", "ruff_format"}:
            command["argv"].append(REPAIRED)
        if command["name"] == "coverage_report":
            command["argv"].append("--include=" + ",".join(str(ROOT / p) for p in OWNED))
    return commands


def main(argv: list[str] | None = None) -> int:
    """Adapt only the new date; the shipped runner still owns deadlines and publication."""
    progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--coverage-custody":
        return retain_shards(Path(args[1])) if len(args) == 2 else 1
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    _, remaining = parser.parse_known_args(args)
    if "--cold-replay" not in remaining:
        os.environ.pop("CARNOT_8379_EXTENSION", None)
    with (
        bindings(),
        patch.multiple(
            runner,
            e=sys.modules[__name__],
            plan=plan,
            audit=audit,
            SCRATCH=Path.home() / ".cache/carnot-exp8405-private",
        ),
    ):
        return int(runner.main(remaining))
