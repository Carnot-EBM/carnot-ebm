"""REQ-REPORT-8355: qualify readers without manufacturing supervisor outcomes.

The qualified empty frontier is still useful evidence. Current execution and
new game support remain separate so a working reader cannot imply learning.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace
from typing import Any, cast

from carnot.reporting import arc_supervisor_artifact_8328 as prior_artifact
from carnot.reporting import arc_supervisor_execution_8328 as prior_execution
from carnot.reporting import arc_supervisor_frontier_8342 as prior
from carnot.reporting import v720_frozen_input_contract as authority
from carnot.reporting.arc_supervisor_v689_delta import operand
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8355_v720_arc_supervisor_frontier"
CLI = "scripts/experiments/" + NAME + ".py"
OUTPUT = ROOT / "results" / (NAME + ".json")
TEST = "tests/python/test_arc_supervisor_frontier_8355.py"
OWNED = ["python/carnot/reporting/arc_supervisor_frontier_8355.py", CLI]
MODEL_SPECS: list[Json] = []
BUDGET = prior.BUDGET
json_document = prior.json_document
HISTORICAL = [ROOT / "results" / (n + ".json") for n in [prior_artifact.NAME, prior.NAME]]


def progress(phase: str) -> None:
    """Flush boundaries because cached aggregation must remain observable too."""
    print(f"[exp8355] phase={phase}", flush=True)


def bind(original: Callable[..., Any], **overrides: Any) -> Callable[..., Any]:
    """Reuse existing statements while binding the exact current invocation."""
    function = cast(FunctionType, original)
    names = [
        "NAME",
        "CLI",
        "OUTPUT",
        "TEST",
        "OWNED",
        "BUDGET",
        "bind",
        "build",
        "replay",
        "plan",
        "controls",
        "publish",
    ]
    scope = dict(function.__globals__) | {key: globals()[key] for key in names} | overrides
    return cast(
        Callable[..., Any],
        FunctionType(
            function.__code__, scope, function.__name__, function.__defaults__, function.__closure__
        ),
    )


def historical() -> tuple[list[Json], dict[str, str]]:
    """Preserve failed outcomes and diagnose stale replay independently of readiness."""
    rows: list[Json] = []
    hashes: dict[str, str] = {}
    for path in HISTORICAL:
        digest = sha256_file(path)
        value = json_document(path)
        side = path.parent / "raw" / path.stem / "validators" / (digest[7:] + ".json")
        bound = read_bound_sidecar(path, side)
        terminal = Path(value["terminal_validation_sidecar_path"])
        report = json_document(terminal)
        if (
            bound["report"]["passed"] is not True
            or bound["primary_path"] != str(path)
            or report["publication"]["primary_sha256"] != digest
        ):
            raise ValueError("historical_terminal_bytes")
        work = json_document(Path(value["work_reference"]["path"]))
        drift = [
            p
            for p, d in work["source_artifact_hashes"].items()
            if not Path(p).is_file() or sha256_file(Path(p)) != d
        ]
        rows.append(
            dict(
                path=str(path),
                sha256=digest,
                verdict_class=value["verdict_class"],
                honest_verdict=value["honest_verdict"],
                authenticated=True,
                readiness_imported=False,
                replay_source_drift=drift,
                original_terminal_replay_passed=all(
                    c["passed"] for c in report["checks"] if c["name"] == "terminal_cold"
                ),
            )
        )
        hashes.update(
            {
                str(p): sha256_file(p)
                for p in [path, side, terminal, Path(value["work_reference"]["path"])]
            }
        )
    return rows, hashes


def measure(raw: Path, private: Path, precondition_failures: list[Json] | None = None) -> Json:
    """Check natural source custody; missing history stays a named external block."""
    work = cast(
        Json,
        bind(prior.measure, authority=authority, progress=progress)(
            raw, private, precondition_failures
        ),
    )
    work["historical_dispositions"] = []
    try:
        work["historical_dispositions"], hashes = historical()
        paths = [
            ROOT / "python/carnot/reporting/v720_frozen_input_contract.py",
            ROOT / "openspec/change-proposals/research-roadmap-v719-preserved-20261009.md",
            ROOT / "results/raw" / NAME / "pre_repair/authenticated_historical_mypy.json",
            ROOT / "results/raw" / NAME / "pre_repair/historical_replay_diagnostic.json",
        ]
        hashes.update({str(p): sha256_file(p) for p in paths})
        for index, (label, digest) in enumerate(hashes.items()):
            saved = raw / "inputs" / (f"v720-{index}-" + digest[7:] + ".bin")
            shutil.copyfile(label, saved)
            work["source_artifact_hashes"][label] = digest
            work["snapshots"][str(saved)] = digest
    except (OSError, ValueError, KeyError) as error:
        path = (
            Path(error.filename) if isinstance(error, OSError) and error.filename else HISTORICAL[0]
        )
        work["failures"].append(
            dict(
                operand(
                    path,
                    "historical_authenticated_bytes",
                    True,
                    str(error) if path.exists() else None,
                    sha256_file(path) if path.is_file() else None,
                ),
                passed=False,
            )
        )
    return work


reader = SimpleNamespace(
    **(vars(prior.reader) | dict(authority=authority, progress=progress, measure=measure))
)
artifact = SimpleNamespace(
    **(
        vars(prior.artifact)
        | dict(m=reader, NAME=NAME, CLI=CLI, OUTPUT=OUTPUT, TEST=TEST, OWNED=OWNED)
    )
)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualified readers earn readiness; only actual new receipts earn outcome support."""
    value = cast(Json, bind(prior_artifact.build, m=reader)(work, receipts, raw, output))
    value.update(
        experiment_id=8355,
        task_id="exp8355-arc-supervisor-frontier",
        milestone="2026.10.720",
        current_game_calls=0,
        current_model_calls=0,
        solve_credit_claimed=False,
        historical_dispositions=work["historical_dispositions"],
    )
    value["field_principles"].update(
        {
            k: "Preserve authenticated history separately from zero current game and model work."
            for k in [
                "current_game_calls",
                "current_model_calls",
                "solve_credit_claimed",
                "historical_dispositions",
            ]
        }
    )
    return value


def replay(value: Json) -> bool:
    """Recompute history as well as native joins so rehashed invented outcomes fail."""
    try:
        work = json_document(Path(value["work_reference"]["path"]))
        if work["historical_dispositions"] and historical()[0] != work["historical_dispositions"]:
            return False
        return bool(bind(prior_artifact.replay, m=reader)(value))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def plan(private: Path) -> list[Json]:
    """Keep existing strict checks and restrict coverage to newly owned statements."""
    specs = cast(list[Json], bind(prior_execution.plan, a=artifact, m=reader)(private))
    selected = "current_reader_and_replay or real_cli_and_failure or block_and_rehashed_history"
    specs[0]["argv"] += ["-k", selected]
    specs.append(
        dict(
            name="frozen_reader_assertions",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                TEST,
                "-k",
                "not (" + selected + ")",
                "--basetemp=" + str(private / "frozen-readers"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        )
    )
    next(s for s in specs if s["name"] == "strict_mypy")["argv"] += [*prior.OWNED]
    return specs


def controls(value: Json, raw: Path) -> list[Json]:
    """Real cold children distinguish valid bytes from altered and rehashed claims."""
    return cast(list[Json], bind(prior_execution.controls, m=reader)(value, raw))


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """Use unchanged typed findings and atomic publication, including honest failures."""
    bind(prior_execution.publish, m=reader)(value, work, output, raw)


def run(output: Path, private: Path, *, fixture: bool = False) -> int:
    """Qualify execution before inspecting the natural frontier once, without game calls."""
    return int(bind(prior_execution.run, a=artifact, m=reader)(output, private, fixture=fixture))
