"""REQ-REPORT-8468: fresh public observations do not grant hidden-game credit.

Only narrow invocation bindings change the qualified collector. Original policy,
SDK recording, supervisor assertions and publication consumers remain intact.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import arc_supervisor_live_panel_8384 as base
from carnot.reporting import arc_supervisor_support_evidence_8439 as evidence
from carnot.reporting import v728_contract_methods as authority
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v721_capstone_evidence import freeze as snapshot, frozen_inputs

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK = "experiment_8468_v728_arc_generalization_panel", "exp8468-arc-generalization-panel"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_generalization_panel_8468.py"
OUTPUT = ROOT / "results" / (NAME + ".json")
OWNED = ["python/carnot/reporting/arc_generalization_panel_8468.py", CLI]
TASK_PIN = "sha256:0ea9a44288af44fc5ba41cb1c88532327775703af357d060aea83f7ad16b2b04"
MODEL_SPECS: list[Json] = []
INPUTS = [
    *base.INPUTS[:11],
    authority.ACTIVE,
    evidence.CANDIDATE,
    "python/carnot/reporting/arc_supervisor_support_8439.py",
    "python/carnot/reporting/arc_supervisor_support_evidence_8439.py",
    "python/carnot/reporting/arc_panel_qualification_8412.py",
    "python/carnot/reporting/arc_supervisor_live_panel_8384.py",
    "python/carnot/agentic/arc_live_panel_runtime_8384.py",
    "python/carnot/agentic/arc_generalization_runtime.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "results/experiment_4629_live_frame_change_cnn.pt",
    *OWNED,
    TEST,
]
runtime, sdk = base.runtime, base.sdk


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so bounded science and validation cannot appear stalled."""
    print(f"[exp8468] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Invocation aliases reuse shipped assertions without rewriting historical producers."""
    with (
        patch.multiple(
            base,
            NAME=NAME,
            TASK=TASK,
            TASK_PIN=TASK_PIN,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            INPUTS=INPUTS,
            build=build,
            progress=progress,
        ),
        patch.object(base.authority_module, "authority", authority.authority),
    ):
        yield


def gate(check: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name missing operands separately from observed zero measurements."""
    return dict(
        check=check,
        upstream=TASK,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def preconditions(private: Path, root: Path) -> tuple[list[Json], Json, Json]:
    """Authenticate complete current authority separately from the sealed original panel."""
    with bindings():
        checks, hashes, resources = base.preconditions(private, root)
    try:
        evidence.producer(root)
    except (OSError, ValueError, KeyError) as error:
        checks.append(
            gate(
                "historical_producer",
                root / evidence.CANDIDATE,
                "original_validated_authority",
                evidence.CANDIDATE_PIN,
                str(error),
            )
        )
    resources.update(measurement_cap_s=3600, validation_reserve_s=900, aggregate_panel_cap_s=2400)
    if checks:
        resources["task_sha256"] = None
    return checks, hashes, resources


def freeze(roster: list[str], registry: list[Json]) -> Json:
    """Reuse only sealed metadata; unavailable selected games cannot be replaced."""
    shipped = evidence.producer(ROOT)["frozen_panel"]["roster"]
    frozen = base.freeze(shipped, registry)
    games = frozen["roster"][:4]
    if len(games) != 4:
        raise ValueError("four_shipped_games_required")
    return dict(
        frozen,
        selected_games=games,
        installed_roster=sorted(roster),
        missing_games=[g for g in games if g not in roster],
        descriptive_old_subset=games[:2],
        aggregate_cap_s=2400,
        units=[
            dict(episode_id=f"{g}:{s}:{a}", game=g, seed=s, arm=a, max_actions=512, max_seconds=120)
            for g in games
            for s in [11, 22]
            for a in ["off", "on"]
        ],
    )


def plan(private: Path) -> list[Json]:
    """Freeze serial owned coverage, real private consumers and wrapper parity."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(commands, "m", SimpleNamespace(ROOT=ROOT, OWNED=OWNED, TEST=TEST)):
        specs = list(commands.manifest(private))
    specs[0]["deadline"] = 900
    next(s for s in specs if s["name"] == "coverage_combine")["argv"].append("--keep")
    for name, tests in [
        (
            "private_E2E011_017_wrapper",
            [
                "test_arc_decision_telemetry.py",
                "test_arc_supervisor_delta_7874.py",
                "test_arc_submitted_agent_parity.py",
            ],
        ),
        ("private_E2E021", ["test_restricted_decision_audit_8210.py"]),
    ]:
        specs.insert(
            1,
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
                    *("tests/python/" + t for t in tests),
                    "--basetemp=" + str(private / name),
                ],
                deadline=300,
                expected=0,
                scope="owned",
            ),
        )
    return specs


def build(work: Json, receipts: list[Json], raw: Path, output: Path, *, seal: bool = True) -> Json:
    """Execution qualification and redirect support are independent engineering claims."""
    work["run_date"] = "20261013"
    value = _BUILD(work, receipts, raw, output, seal=seal)
    rows = value["rows"]
    if work.get("panel") is None:
        rows.extend(
            dict(
                episode_id=f"missing:{g}:{seed}:{arm}",
                game=None,
                seed=seed,
                arm=arm,
                status="unstarted",
                absolute_metric=None,
                action_count=None,
                missing_reason="external_precondition_or_qualification",
                censored=True,
            )
            for g in [2, 3]
            for seed in [11, 22]
            for arm in ["off", "on"]
        )
        value["censored_count"] = 16
    summary = evidence.summarize(work["rows"])
    ready = int(
        value["required_checks_passed"]
        and value["verdict_class"] == "null"
        and len(work["rows"]) == 16
        and all(r.get("model_tripwire_passed") for r in rows)
    )
    games = (work.get("panel") or {}).get("selected_games", [])
    paired = []
    for game in games:
        for seed in [11, 22]:
            arms = {r["arm"]: r for r in rows if r["game"] == game and r["seed"] == seed}
            off, on = arms["off"], arms["on"]
            paired.append(
                dict(
                    game=game,
                    seed=seed,
                    arms=arms,
                    peak_level_delta_on_minus_off=on["peak_level"] - off["peak_level"]
                    if off.get("peak_level") is not None and on.get("peak_level") is not None
                    else None,
                    missing_reason=None
                    if off.get("peak_level") is not None and on.get("peak_level") is not None
                    else "missing_paired_observation",
                )
            )
    value.update(
        experiment_id=8468,
        task_id=TASK,
        milestone="2026.10.728",
        honest_verdict=f"complete_{value['verdict_class']}_v728_arc_generalization_panel",
        intended_count=16,
        panel_execution_ready_score=ready,
        arc_panel_ready_score=ready,
        arc_receipt_ready_score=ready,
        arc_support_ready_score=int(
            ready and any(cell["supported"] for cell in summary["overlap_by_arm"].values())
        ),
        sample_size_budget=dict(
            games=4,
            seeds=[11, 22],
            arms=["off", "on"],
            intended_episodes=16,
            max_actions=512,
            max_seconds=120,
            aggregate_seconds=2400,
            supervisor_window=120,
            independent_unit="public_game",
            hidden_games=0,
        ),
        per_episode_rows=rows,
        per_game_results=summary["per_game_results"],
        redirect_outcome_rows=summary["supervisor_outcome_rows"],
        censoring_counts=summary["censoring_counts"],
        overlap_by_arm=summary["overlap_by_arm"],
        refinement_status="supported"
        if ready and any(c["supported"] for c in summary["overlap_by_arm"].values())
        else "blocked_insufficient_resolved_overlap",
        arm_refinement_authorized=False,
        paired_progress_rows=paired,
        descriptive_old_subset=[r for r in paired if r["game"] in games[:2]],
        original_validated_hash=evidence.CANDIDATE_PIN,
        current_observed_hash=work.get("current_observed_hash"),
        new_level_solves=0,
        policy_model_calls=0,
        source_input_receipts=work.get("source_input_receipts", []),
        reader_substrate="aggregation_from_upstream_artifacts",
        numerical_head_substrate="verifier_ensemble_against_cached_candidates",
        official_score_availability=dict(
            available=False,
            value=None,
            missing_reason="offline_SDK_frame_does_not_supply_official_competition_score",
        ),
        action_provenance="actual_make_carnot_agent_E3AgentPolicy_choose_action_SDK_transition",
        historical_limits=dict(
            V717_H1=-0.00390625,
            V717_H2=0,
            V726_admission_audit="null",
            kernel_service_percent=0.093,
            persistence_service_percent=96.44,
            V727="six blocked primaries and eight skipped slots",
        ),
    )
    value["acceptance_gates"].update(
        fixed_panel=bool(ready), redirect_support=bool(value["arc_support_ready_score"])
    )
    value["field_principles"].update(
        {
            k: "Bind to this invocation and sealed primitive operands; public engineering evidence grants no hidden-game benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        intended_count="Freeze16 episodes before observations, with no data-dependent replacement.",
        milestone="Bind fresh model-free work to complete activated V728 authority.",
        run_date="Authorized invocation date20261013.",
        panel_execution_ready_score="All16 live receipts and required validation qualify execution only.",
        arc_support_ready_score="Require3 shared games and5 resolved outcomes per shared arm cell.",
        paired_progress_rows="Compare all eight game/seed pairs even when no redirect fires.",
        censored_count="Episode completion and bounded observation horizon can overlap.",
        per_game_results="Game/arm cells retain resolved and right-censored events separately.",
        official_score_availability="Do not invent an official score from diagnostic levels.",
        policy_model_calls="Guarded existing model-free policy makes zero current LLM calls.",
        new_level_solves="Registry-prechecked public progress receives no new solve credit.",
        historical_model_provenance="Cached generator and common numeric head provenance never counts as current LLM generation.",
    )
    return dict(value)


@contextmanager
def original_bindings() -> Iterator[None]:
    """Allow shipped replay to call the new reducer while preserving original functions."""
    with bindings():
        yield


_BUILD, _REPLAY = base.build, base.replay


def replay(value: Json) -> bool:
    """Fresh reduction and independent authority reject rehashed summaries and absent bytes."""
    try:
        with original_bindings():
            if not _REPLAY(value):
                return False
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        refs = work.get("source_input_receipts", [])
        if any(sha256_file(Path(r["snapshot_path"])) != r["sha256"] for r in refs):
            return False
        if work["preconditions_checked"].get("task_sha256") == TASK_PIN:
            with TemporaryDirectory(prefix="exp8468-replay-", dir="/var/tmp") as directory:
                with frozen_inputs(refs, Path(directory) / "writes"):
                    contract = authority.authority(ROOT, Path(directory) / "authority")
                    task = next(t for t in contract["tasks"] if t["id"] == TASK)
                    if not contract["activated"] or canonical_hash(task) != TASK_PIN:
                        return False
                    registry = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_bytes())[
                        "games"
                    ]
                    if (
                        work.get("panel")
                        and freeze(work["panel"]["installed_roster"], registry) != work["panel"]
                    ):
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def controls(value: Json, raw: Path) -> list[Json]:
    """Actual fresh children test valid, missing, error and self-rehashed primitive candidates."""
    with bindings():
        return list(base.controls(value, raw))


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """Unchanged validators retain failures before checked terminal bytes become visible."""
    with bindings():
        base.publish(value, work, output, raw)


def run(output: Path, private: Path, *, private_e2e: bool = False) -> int:
    """Freeze before observing and reserve validation time after bounded CPU collection."""
    started = time.monotonic()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_before")
    specs = [] if private_e2e else plan(private)
    atomic_json(
        raw / "command_manifest.json",
        dict(
            commands=specs,
            owned=OWNED,
            task_cap_s=4800,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            heartbeat_s=30,
            MODEL_SPECS=[],
            seeds=[11, 22],
            arms=["off", "on"],
        ),
    )
    checks, hashes, resources = preconditions(private, ROOT)
    refs = [
        snapshot(Path(name), raw / "inputs", digest) for name, digest in hashes.items() if digest
    ]
    panel = None
    if private_e2e:
        checks.append(
            gate("private_constructed_control", private, "research_observations", True, None)
        )
    else:
        try:
            progress("sdk_metadata_before")
            arcade = sdk()
            registry = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_bytes())["games"]
            panel = freeze([str(g.game_id) for g in arcade.available_environments], registry)
            atomic_json(raw / "frozen_panel.json", panel)
            progress("sdk_metadata_after", 4, 0)
            for game in panel["missing_games"]:
                checks.append(
                    gate(
                        "installed_frozen_game", ROOT / "environment_files", game, "installed", None
                    )
                )
        except (ImportError, OSError, ValueError) as error:
            checks.append(
                gate(
                    "sdk_access", ROOT / "environment_files", "four_frozen_games", True, str(error)
                )
            )
    progress("qualification_before")
    receipts = execute(specs, raw / "checks")
    for path in private.iterdir():
        if path.is_file() and path.name.startswith((".coverage", "coverage")):
            target = raw / "coverage" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    rows = []
    collection_started = time.monotonic()
    if (
        panel
        and not checks
        and all(r["passed"] for r in receipts)
        and resources.get("task_sha256") == TASK_PIN
    ):
        progress("live_panel_before", 0, 16)
        for index, unit in enumerate(panel["units"]):
            if (
                time.monotonic() - started + 135 > 3600
                or time.monotonic() - collection_started + 135 > 2400
            ):
                checks.append(
                    gate("measurement_deadline", raw, "within_measurement_budget", True, False)
                )
                break
            episode_raw = raw / "episodes" / str(index)
            operand = raw / f"unit-{index}.json"
            atomic_json(operand, unit)
            receipt = child(
                "episode_" + str(index),
                [
                    sys.executable,
                    "-u",
                    str(ROOT / CLI),
                    "--episode",
                    str(operand),
                    "--raw",
                    str(episode_raw),
                ],
                raw / "episode_logs",
                deadline=135,
                scope="measurement",
            )
            terminal_path = episode_raw / "episode.json"
            terminal = json.loads(terminal_path.read_bytes()) if terminal_path.exists() else None
            rows.append(base.reduce_episode(unit, terminal, episode_raw))
            if not receipt["passed"]:
                receipt["scope"] = "owned"
            receipts.append(receipt)
            progress("live_panel_progress", index + 1, 15 - index)
        progress("live_panel_after", len(rows), 16 - len(rows))
    try:
        source = evidence.producer(ROOT)
    except (OSError, ValueError, KeyError):
        source = {}
    observed = ROOT / evidence.UPSTREAM
    work = dict(
        panel=panel,
        rows=rows,
        failures=checks,
        source_artifact_hashes=hashes,
        source_input_receipts=refs,
        raw_shard_hashes={str(p): sha256_file(p) for p in raw.rglob("*") if p.is_file()},
        code_config_hashes={n: sha256_file(ROOT / n) for n in OWNED},
        preconditions_checked=resources,
        historical_model_provenance=dict(
            imported=source.get("historical_model_provenance", []),
            numeric_head=dict(
                path="results/experiment_4629_live_frame_change_cnn.pt",
                sha256=hashes.get(str(ROOT / "results/experiment_4629_live_frame_change_cnn.pt")),
                per_game_memory=False,
                current_llm_calls=0,
            ),
        ),
        current_observed_hash=sha256_file(observed) if observed.is_file() else None,
        phase_spans=[
            dict(phase="qualification", duration_s=collection_started - started),
            dict(phase="live_panel", duration_s=time.monotonic() - collection_started),
        ],
        duration_s=time.monotonic() - started,
        run_date="20261013",
    )
    value = build(work, receipts, raw, output)
    progress("cold_controls_before")
    receipts.extend(controls(value, raw))
    value = build(work, receipts, raw, output)
    progress("terminal_validation_before")
    publish(value, work, output, raw)
    progress("terminal_publication_after")
    return 0
