"""REQ-REPORT-8987-ADMISSION: reuse bounded validation and atomic publication mechanics."""

from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from carnot.reporting import spline_development_admission_8987 as d
from carnot.reporting import spline_basis_runner_8986 as base
from carnot.reporting import matched_corpus_runner_8949 as shared

Json = dict[str, Any]
_plan = shared.validation_plan
_freeze = base.freeze


def validation_plan(private: Path, *, producer: ModuleType | None = None) -> Any:
    """Freeze selectors only in pytest commands and measure only newly owned statements."""
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
    """Reuse original resource checks, retained inputs and phase clocks before measurement."""
    with patch.object(base, "d", d):
        work = _freeze(root, raw)
    for name in [
        *d.base.OWNED,
        "python/carnot/reporting/relational_methods_8959.py",
        "python/carnot/reporting/graph_source_8931.py",
    ]:
        work["sources"].append(
            dict(d.retain(d.reference(d.ROOT / name), raw), executed_source=True)
        )
    return work


def replay(path: Path) -> Json:
    """Rebuild summaries from immutable operands without running optimization or generation."""
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
            for source, pin, payload in [
                (d.INPUT, d.INPUT_PIN, evidence["spline_methods"]),
                (d.base.PROTOCOL, d.base.PROTOCOL_PIN, evidence["protocol"]),
                (d.base.CONTEXT, d.base.CONTEXT_PIN, evidence["context"]),
                (d.historical.INPUT, d.historical.PIN, evidence["producer"]),
            ]:
                ref = next(
                    r
                    for r in evidence["references"]
                    if r.get("source_path") == str(d.ROOT / source)
                )
                if ref["sha256"] != pin or json.loads(Path(ref["path"]).read_bytes()) != payload:
                    raise ValueError("pinned_operand_drift")
            if (
                d.historical.load(evidence, evidence["producer"]["work_reference"])
                != evidence["work"]
            ):
                raise ValueError("pinned_work_drift")
            d.verify_result(evidence, work["result"])
        d.progress("after_cold_replay", 1, 0)
        return dict(passed=True, replay_passed=True)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        d.progress("cold_replay_refused", 0, 1)
        return dict(passed=False, replay_passed=False, reason=str(error))


@contextmanager
def binding() -> Iterator[None]:
    """Use the existing checked CLI with this experiment's explicit identity and replay."""
    with (
        patch.object(base, "d", d),
        patch.object(base, "replay", replay),
        patch.object(base, "freeze", freeze),
        patch.object(shared, "validation_plan", validation_plan),
    ):
        yield


def main(argv: list[str] | None = None) -> int:
    """The reusable runner retains private CLI, validation receipts and unchanged scanners."""
    with binding():
        return base.main(argv)
