"""REQ-REPORT-8988-CAUSAL: reuse bounded checks and checked atomic publication."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import time
from types import ModuleType
from typing import Any
from unittest.mock import patch

from carnot.reporting import online_spline_methods_8988 as d
from carnot.reporting import spline_basis_runner_8986 as base
from carnot.reporting.spline_basis_methods_8986 import preflight
from carnot.reporting import matched_corpus_runner_8949 as shared

Json = dict[str, Any]
_plan = shared.validation_plan


def validation_plan(private: Path, *, producer: ModuleType | None = None) -> Any:
    """Measure only owned statements; put selectors only into pytest and spec checks."""
    return [
        replace(
            c,
            argv=(
                *c.argv[: c.argv.index("pytest") + 1],
                "--noconftest",
                "-s",
                *c.argv[c.argv.index("pytest") + 1 :],
            ),
        )
        if "pytest" in c.argv
        else replace(c, argv=(c.argv[0], "--noconftest", *c.argv[1:]))
        if c.name in {"e2e018", "e2e021"}
        else c
        for c in _plan(private, producer=d)
    ]


def freeze(root: Path, raw: Path) -> Json:
    """Authenticate exact cached inputs, resources and dependencies before private compute."""
    started = time.monotonic()
    d.progress("before_preconditions", 0, 1)
    evidence = d.authenticate(root, raw)
    preconditions = preflight(raw, evidence)
    preconditions["required_network"] = (
        "none: all operands are retained cached bytes; no model load or generation"
    )
    authenticated = time.monotonic()
    d.progress("after_preconditions", int(not evidence["failures"]), len(evidence["failures"]))
    sources = [
        dict(
            d.retain(d.reference(d.ROOT / name), raw),
            executed_source=name in d.OWNED or name == d.PROTOCOL,
        )
        for name in [
            *d.OWNED,
            *d.TESTS,
            d.PROTOCOL,
            "python/carnot/autoresearch/spline_basis_8986.py",
            "python/carnot/reporting/spline_basis_runner_8986.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/current_work_receipt.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "openspec/capabilities/research-reporting/spec.md",
        ]
    ]
    result, failure = {}, None
    if not evidence["failures"]:
        try:
            result = d.evaluate(evidence, raw)
        except ValueError as error:
            failure = str(error)
            d.progress("measurement_refused", 0, 1)
    ended = time.monotonic()
    return dict(
        evidence=evidence,
        result=result,
        sources=sources,
        raw=str(raw),
        preconditions=preconditions,
        duration_s=ended - started,
        phase_spans=[
            dict(phase="preconditions", duration_s=authenticated - started),
            dict(phase="private_methods", duration_s=ended - authenticated),
        ],
        global_test_debt={},
        raw_references=[],
        owned_failure=failure,
    )


def replay(path: Path) -> Json:
    """Verify retained bytes and independently reconstruct states and primitive summaries."""
    d.progress("before_cold_replay")
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if d.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("retained_bytes_drift")
            if (
                ref.get("executed_source")
                and d.reference(Path(ref["source_path"]))["sha256"] != ref["sha256"]
            ):
                raise ValueError("executed_bytes_drift")
        raw = Path(value["work_reference"]["path"]).parent
        work = json.loads((raw / "work.json").read_bytes())
        validation = json.loads((raw / "validation.json").read_bytes())
        work["raw_references"] = value["raw_shard_hashes"]
        with patch.object(base, "d", d):
            if base.artifact(work, validation, value["fixture_only"]) != value:
                raise ValueError("aggregate_drift")
        evidence = work["evidence"]
        if not evidence["failures"]:
            ref = next(
                r for r in evidence["references"] if r.get("source_path") == str(d.ROOT / d.INPUT)
            )
            if (
                ref["sha256"] != d.INPUT_PIN
                or json.loads(Path(ref["path"]).read_bytes()) != evidence["producer"]
            ):
                raise ValueError("pinned_operand_drift")
            if d.evaluate(evidence, raw, replay=True) != work["result"]:
                raise ValueError("reduction_drift")
        d.progress("after_cold_replay", 1, 0)
        return dict(passed=True, replay_passed=True)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        d.progress("cold_replay_refused", 0, 1)
        return dict(passed=False, replay_passed=False, reason=str(error))


@contextmanager
def binding() -> Iterator[None]:
    """Bind the existing dated CLI to this experiment's producer and replay."""
    with (
        patch.object(base, "d", d),
        patch.object(base, "freeze", freeze),
        patch.object(base, "replay", replay),
        patch.object(shared, "validation_plan", validation_plan),
    ):
        yield


def main(argv: list[str] | None = None) -> int:
    """The shared runner retains frozen argv, normal exits, log hashes and terminal validators."""
    with binding():
        code: int = base.main(argv)
        return code
