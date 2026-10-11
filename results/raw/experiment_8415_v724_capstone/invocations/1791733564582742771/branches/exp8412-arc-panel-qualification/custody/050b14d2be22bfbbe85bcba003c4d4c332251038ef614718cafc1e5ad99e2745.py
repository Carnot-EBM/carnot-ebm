"""REQ-REPORT-8370: count new supervisor outcomes after a qualified frontier.

A working reader can observe no new evidence. Keep execution readiness separate
from outcome support so an empty ledger cannot imply a learning benefit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil
import time
from types import FunctionType, SimpleNamespace
from typing import Any, cast

from carnot.reporting import arc_supervisor_frontier_8355 as prior
from carnot.reporting import arc_supervisor_frontier_8328 as native
from carnot.reporting import v721_contract_methods as authority
from carnot.reporting.arc_supervisor_v688_receipts import event_order
from carnot.reporting.arc_supervisor_v689_delta import operand
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8370_v721_arc_outcome_delta"
CLI = "scripts/experiments/" + NAME + ".py"
OUTPUT = ROOT / "results" / (NAME + ".json")
TEST = "tests/python/test_arc_outcome_delta_8370.py"
OWNED = ["python/carnot/reporting/arc_outcome_delta_8370.py", CLI]
FRONTIER = prior.OUTPUT
PIN = "sha256:a93696d2e2b7cf9ee7a676652b89f02f8d93872cf346de9fccd2bf884586f394"
READERS = [*native.READERS, *prior.OWNED]
MODEL_SPECS: list[Json] = []
BUDGET = dict(prior.BUDGET, minimum_overlapping_games=8, minimum_resolved_events_per_arm=20)
json_document = prior.json_document


def progress(phase: str) -> None:
    """Flush boundaries so bounded evidence reading stays visible to the parent."""
    print(f"[exp8370] phase={phase}", flush=True)


def bind(original: Callable[..., Any], **overrides: Any) -> Callable[..., Any]:
    """Reuse qualified statements with explicit current identities and reader seams."""
    function = cast(FunctionType, original)
    names = [
        "NAME",
        "CLI",
        "OUTPUT",
        "TEST",
        "OWNED",
        "BUDGET",
        "build",
        "replay",
        "plan",
        "controls",
        "publish",
        "progress",
        "FRONTIER",
        "PIN",
        "READERS",
        "authenticate",
        "inspect",
        "reader",
        "authority",
    ]
    adapted = FunctionType(
        function.__code__,
        dict(function.__globals__) | {key: globals()[key] for key in names} | overrides,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    adapted.__kwdefaults__ = function.__kwdefaults__
    return cast(Callable[..., Any], adapted)


def authenticate(path: Path) -> Json:
    """Exact prior bytes and both terminals qualify imports without replaying old science."""
    if sha256_file(path) != PIN:
        raise ValueError("qualification_bytes")
    value = json_document(path)
    side = path.parent / "raw" / path.stem / "validators" / (PIN[7:] + ".json")
    bound = read_bound_sidecar(path, side)
    if bound.get("report", {}).get("passed") is not True or bound.get("primary_path") != str(
        path.absolute()
    ):
        raise ValueError("qualification_terminal")
    if value.get("arc_reader_ready_score") != 1 or value.get("required_checks_passed") is not True:
        raise ValueError("reader_not_qualified")
    for label in READERS:
        expected = value["code_config_hashes"].get(label) or value["source_artifact_hashes"].get(
            str(ROOT / label)
        )
        if expected is None or sha256_file(ROOT / label) != expected:
            raise ValueError("reader_code_changed:" + label)
    report = json_document(Path(value["terminal_validation_sidecar_path"]))
    if report.get("publication", {}).get("primary_sha256") != PIN or not report.get("passed"):
        raise ValueError("terminal_report")
    return cast(Json, value)


def summarize(rows: list[Json], frontier: Json) -> Json:
    """Keep every event while requiring shared game support for descriptive proposals."""
    value = native.reader.summarize(rows, frontier)
    supported = (
        value["arm_overlap_games"] >= 8
        and len(value["shared_arms"]) >= 2
        and all(
            value["per_arm_results"][arm]["resolved_by_levelup"] >= 20
            for arm in value["shared_arms"]
        )
    )
    if not supported:
        value.update(
            proposed_arm_change=None,
            proposed_generalization_change=None,
            descriptive_leave_one_game_out=[],
            support_status="insufficient_support",
        )
    value["selection_propensity"] = dict(status="unknown", value=None)
    value["reopen_condition"] = (
        "New authenticated post-Exp8355 receipt IDs with environment outcomes. Descriptive comparison requires eight shared games, twenty resolved events per arm and five firings per game-arm cell."
    )
    return cast(Json, value)


def inspect(locator: Path, previous: Json, raw: Path) -> Json:
    """Reuse native registry and environment joins with the exact prior finished-at cutoff."""
    parser = SimpleNamespace(
        **(
            vars(native.native)
            | dict(
                summarize=summarize,
                event_order=lambda row, cutoff, current: event_order(row, cutoff, "20261010"),
            )
        )
    )
    return cast(
        Json,
        bind(
            native.inspect,
            native=parser,
            event_order=lambda row, cutoff, current: event_order(row, cutoff, "20261010"),
        )(locator, previous, raw),
    )


def measure(raw: Path, private: Path, precondition_failures: list[Json] | None = None) -> Json:
    """Authenticate authority and private resources before inspecting natural evidence once."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_before")
    hashes: dict[str, str] = {}
    checks = list(precondition_failures or [])
    previous: Json = {}
    delta = summarize([], {})
    path, field = private, "private_scratch"
    try:
        probe = private / "exp8370-private-probe"
        probe.write_bytes(b"private scratch")
        if (
            private.resolve().is_relative_to(ROOT)
            or shutil.disk_usage(private).free < 10_000_000
            or probe.read_bytes() != b"private scratch"
        ):
            raise ValueError("private_scratch")
        if checks:
            raise ValueError("missing_tools")
        path, field = FRONTIER, "arc_reader_ready_score"
        previous = authenticate(FRONTIER)
        hashes[str(FRONTIER)] = PIN
        terminal = Path(previous["terminal_validation_sidecar_path"])
        bound = FRONTIER.parent / "raw" / FRONTIER.stem / "validators" / (PIN[7:] + ".json")
        path, field = ROOT / authority.ACTIVE, "activated"
        contract = authority.authority(ROOT, raw / "authority")
        checks.extend(contract["gate_check_summary"])
        if not contract["activated"]:
            raise ValueError("current_authority")
        path, field = ROOT / authority.legacy.PROTOCOL, "protocol_sha256"
        if sha256_file(path) != authority.legacy.base.PIN:
            checks.append(
                dict(
                    operand(
                        path, field, authority.legacy.base.PIN, sha256_file(path), sha256_file(path)
                    ),
                    passed=False,
                )
            )
            raise ValueError("frozen_protocol_changed")
        labels = [
            *READERS,
            *OWNED,
            TEST,
            authority.PROTOCOL,
            authority.legacy.PROTOCOL,
            authority.DESIGN,
            authority.ACTIVE,
            "ops/exclusion_manifest.yaml",
            "python/carnot/reporting/v721_contract_methods.py",
            "python/carnot/reporting/arc_supervisor_frontier_8328.py",
            "python/carnot/reporting/arc_supervisor_artifact_8328.py",
            "python/carnot/reporting/arc_supervisor_execution_8328.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/v717_contract_runner.py",
            "python/carnot/reporting/v718_replay_history.py",
            "python/carnot/reporting/v718_replay_runner.py",
            "python/carnot/reporting/primary_publication.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
        for source in [
            *(ROOT / label for label in labels),
            terminal,
            bound,
            raw / "command_manifest.json",
        ]:
            if source.is_file():
                hashes[str(source)] = sha256_file(source)
        locator = raw / "authority_locator.v1.json"
        atomic_json(locator, native.reader.authority.discover())
        progress("preconditions_after_measurement_before")
        delta = inspect(locator, previous, raw / "adapter")
        checks.extend(delta["failures"])
        hashes.update(delta["source_artifact_hashes"])
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, OSError) and error.filename:
            path, field = Path(error.filename), "is_file"
        checks.append(
            dict(
                operand(
                    path,
                    field,
                    True,
                    str(error) if path.exists() else None,
                    sha256_file(path) if path.is_file() else None,
                ),
                passed=False,
            )
        )
    snapshots: dict[str, str] = {}
    for index, (label, digest) in enumerate(hashes.items()):
        saved = raw / "inputs" / (str(index) + "-" + digest[7:] + ".bin")
        saved.parent.mkdir(exist_ok=True)
        shutil.copyfile(label, saved)
        snapshots[str(saved)] = digest
    end = time.monotonic_ns()
    progress("measurement_after")
    return dict(
        delta=delta,
        failures=checks,
        source_artifact_hashes=hashes,
        snapshots=snapshots,
        prior_frontier=previous.get("current_frontier", {}),
        frontier_finished_at=previous.get("finished_at"),
        historical_model_provenance=previous.get("historical_model_provenance", []),
        phase_spans=[
            dict(
                phase="authenticate_and_inspect", started_monotonic_ns=start, ended_monotonic_ns=end
            )
        ],
        duration_s=(end - start) / 1e9,
        private_scratch=str(private),
    )


reader = SimpleNamespace(
    **(
        vars(native)
        | dict(
            authority=authority,
            progress=progress,
            FRONTIER=FRONTIER,
            PIN=PIN,
            READERS=READERS,
            authenticate=authenticate,
            inspect=inspect,
            measure=measure,
            summarize=summarize,
            reader=SimpleNamespace(summarize=summarize),
        )
    )
)
artifact = SimpleNamespace(
    **(
        vars(prior.prior_artifact)
        | dict(m=reader, NAME=NAME, CLI=CLI, OUTPUT=OUTPUT, TEST=TEST, OWNED=OWNED, BUDGET=BUDGET)
    )
)


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """An authenticated empty read earns reader readiness and no scientific support."""
    value = cast(Json, bind(prior.prior_artifact.build, m=reader)(work, receipts, raw, output))
    value.update(
        experiment_id=8370,
        task_id="exp8370-arc-outcome-delta",
        milestone="2026.10.721",
        run_date="20261010",
        current_game_calls=0,
        current_model_calls=0,
        solve_credit_claimed=False,
        prior_frontier=dict(
            path=str(FRONTIER),
            sha256=PIN,
            finished_at=work["frontier_finished_at"],
            current_frontier=work["prior_frontier"],
        ),
        no_change_receipt=dict(
            authenticated=True,
            frontier_sha256=PIN,
            cutoff_finished_at=work["frontier_finished_at"],
            source_artifact_hashes=work["delta"].get("frontier_hashes", {}),
            new_completed_outcome_count=0,
            current_game_calls=0,
            current_model_calls=0,
            arm_change=None,
            fixture_claim_scope=bool(work.get("fixture_claim_scope")),
        )
        if not work["failures"] and not value["completed_count"]
        else None,
    )
    if value["verdict_class"] == "positive" and not value["completed_count"]:
        value.update(
            verdict_class="null", honest_verdict="complete_null_no_completed_supervisor_outcomes"
        )
    value["arc_outcome_support_score"] = int(
        bool(value["arc_reader_ready_score"] and value["completed_count"])
    )
    value["cited_upstream_artifacts"][0]["fields_imported"].append("historical_model_provenance")
    value["field_principles"].update(
        {
            k: "Bind this invocation to qualified prior bytes; zero current calls and no solve credit."
            for k in [
                "current_game_calls",
                "current_model_calls",
                "solve_credit_claimed",
                "prior_frontier",
                "no_change_receipt",
            ]
        }
    )
    return value


def replay(value: Json) -> bool:
    """Native joins and a fresh rebuild reject invented counts even with updated hashes."""
    return bool(bind(prior.prior_artifact.replay, m=reader)(value))


def plan(private: Path) -> list[Json]:
    """Freeze owned coverage and qualified private controls before natural inspection."""
    specs = cast(list[Json], bind(prior.prior_execution.plan, a=artifact, m=reader)(private))
    next(s for s in specs if s["name"] == "qualified_consumers")["name"] = "qualified_ARC_reader"
    for name, files in [
        ("private_E2E018_V721", [authority.TEST + "::test_private_e2e018"]),
    ]:
        specs.append(
            dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *files,
                    "--basetemp=" + str(private / name),
                ],
                expected=0,
                deadline=180,
                scope="owned",
            )
        )
    specs.append(
        dict(
            name="private_disk_scratch",
            argv=["/usr/bin/df", "-T", str(private)],
            expected=0,
            deadline=10,
            scope="owned",
        )
    )
    return specs


def controls(value: Json, raw: Path) -> list[Json]:
    """Use real cold children for valid, negative and rehashed tamper candidates."""
    return cast(list[Json], bind(prior.prior_execution.controls, m=reader)(value, raw))


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """Keep unchanged validators and every finding before atomically exposing bytes."""
    bind(prior.prior_execution.publish, m=reader)(value, work, output, raw)


def run(output: Path, private: Path, *, fixture: bool = False) -> int:
    """Qualify bounded execution before the one natural read, with no model loads."""
    return int(
        bind(prior.prior_execution.run, a=artifact, m=reader)(output, private, fixture=fixture)
    )
