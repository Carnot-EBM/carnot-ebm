"""REQ-REPORT-8398: qualify public transfer without changing the shipped agent.

Current authority is checked independently. Historical games and protocols remain
frozen so a new contract cannot silently replace a difficult panel.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import arc_supervisor_live_panel_8384 as base
from carnot.reporting import v723_contract_methods as authority
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v721_capstone_evidence import freeze as snapshot, frozen_inputs

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK = "experiment_8398_v723_arc_generalization_panel", "exp8398-arc-generalization-panel"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_generalization_panel_8398.py"
OUTPUT = ROOT / "results" / (NAME + ".json")
OWNED = ["python/carnot/reporting/arc_generalization_panel_8398.py", CLI]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:3b683bec2572556b8ca2833aa3988ed47c469f68941781974d024d7e8adb6079"
ORIGINAL = "results/experiment_8384_v722_arc_supervisor_live_panel.json"
ORIGINAL_PIN = "sha256:0fe18c4fd2e507341bc788ca461e0dd194993a67aaae1ce0c1ad71eea1149100"
INPUTS = list(
    dict.fromkeys([*base.INPUTS, ORIGINAL, authority.PROTOCOL, authority.METHODS, *OWNED, TEST])
)
BASE_BUILD, BASE_REPLAY, BASE_PRECONDITIONS, BASE_FREEZE = (
    base.build,
    base.replay,
    base.preconditions,
    base.freeze,
)
runtime, sdk = base.runtime, base.sdk


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real boundaries so bounded collection and synthesis remain observable."""
    print(f"[exp8398] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(check: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """A missing external operand must name its source and never become a measured zero."""
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


@contextmanager
def bindings() -> Iterator[None]:
    """Process-local bindings reuse validated machinery without editing historical producers."""
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


def preconditions(private: Path, root: Path) -> tuple[list[Json], Json, Json]:
    """Bind full current authority, frozen history, actual tools and private disk resources."""
    with bindings():
        checks, hashes, resources = BASE_PRECONDITIONS(private, root)
    for name, pin in [
        (ORIGINAL, ORIGINAL_PIN),
        (authority.PROTOCOL, authority.PROTOCOL_PIN),
        (authority.METHODS, authority.METHODS_PIN),
    ]:
        observed = hashes.get(str(root / name))
        if observed != pin:
            checks.append(
                gate(
                    "frozen_panel_hash" if name == ORIGINAL else "protocol_hash",
                    root / name,
                    "sha256",
                    pin,
                    observed,
                )
            )
    return checks, hashes, resources


def freeze(roster: list[str], registry: list[Json]) -> Json:
    """Reuse the original games even when today's SDK omits one or offers an easier game."""
    original = ROOT / ORIGINAL
    if sha256_file(original) != ORIGINAL_PIN:
        raise ValueError("frozen_panel_hash")
    historical = json.loads(original.read_bytes())["frozen_panel"]
    panel = BASE_FREEZE(historical["roster"], registry)
    if panel["units"] != historical["units"]:
        raise ValueError("frozen_panel_units")
    return dict(
        panel,
        installed_roster=sorted(roster),
        missing_games=[g for g in panel["selected_games"] if g not in roster],
    )


def plan(private: Path) -> list[Json]:
    """Freeze only owned commands; authenticated old global health is reported separately."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(commands, "m", SimpleNamespace(ROOT=ROOT, OWNED=OWNED, TEST=TEST)):
        specs = list(commands.manifest(private))
    specs[0]["deadline"] = 600
    specs.insert(
        1,
        dict(
            name="private_E2E011_017_wrapper",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_arc_decision_telemetry.py",
                "tests/python/test_arc_supervisor_delta_7874.py",
                "tests/python/test_arc_submitted_agent_parity.py",
                "--basetemp=" + str(private / "wrapper"),
            ],
            deadline=300,
            expected=0,
            scope="owned",
        ),
    )
    specs.insert(
        2,
        dict(
            name="historical_wrapper_assertions",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                "-c",
                "from pathlib import Path; from carnot.reporting.arc_generalization_panel_8398 import historical_checks; import sys; sys.exit(historical_checks(Path(sys.argv[1])))",
                str(private / "historical-wrapper"),
            ],
            deadline=300,
            expected=0,
            scope="owned",
        ),
    )
    return specs


def historical_checks(private: Path) -> int:
    """Run unchanged V722 assertions against their original input closure, not today's authority."""
    import pytest

    original = json.loads((ROOT / ORIGINAL).read_bytes())
    saved = {digest: path for path, digest in original["raw_shard_hashes"].items()}
    refs = [
        dict(path=name, sha256=digest, exists=True, snapshot_path=saved[digest])
        for name, digest in original["source_artifact_hashes"].items()
        if digest
    ]
    for ref in refs:
        if sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]:
            raise ValueError("historical_snapshot_hash")
    with frozen_inputs(refs, private / "writes"):
        return int(
            pytest.main(
                [
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    base.TEST,
                    "--basetemp=" + str(private / "tests"),
                ]
            )
        )


def build(work: Json, receipts: list[Json], raw: Path, output: Path, *, seal: bool = True) -> Json:
    """Separate qualified public execution from scientific benefit and historical authority."""
    value = BASE_BUILD(work, receipts, raw, output, seal=seal)
    value.update(
        experiment_id=8398,
        task_id=TASK,
        milestone="2026.10.723",
        honest_verdict=f"complete_{value['verdict_class']}_v723_public_generalization_panel",
        source_input_receipts=work.get("source_input_receipts", []),
    )
    value["field_principles"].update(
        milestone="Bind current V723 authority without repairing V722 history.",
        frozen_panel="Reuse V722 games and budgets; absent installed games keep blocked slots.",
        source_input_receipts="Cold authority checks read hash-bound immutable input snapshots.",
        global_repository_health="Import the authenticated original global timeout separately from owned checks.",
    )
    return dict(value)


def replay(value: Json) -> bool:
    """Recompute rows and current authority from sealed bytes, rather than trusting a rehash."""
    try:
        with bindings():
            if not BASE_REPLAY(value):
                return False
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        refs = work.get("source_input_receipts", [])
        if any(sha256_file(Path(r["snapshot_path"])) != r["sha256"] for r in refs):
            return False
        if work["preconditions_checked"].get("task_sha256") == TASK_PIN:
            with TemporaryDirectory(prefix="exp8398-cold-", dir="/var/tmp") as scratch:
                with frozen_inputs(refs, Path(scratch) / "writes"):
                    current = authority.authority(ROOT, Path(scratch) / "authority")
                    task = next(t for t in current["tasks"] if t["id"] == TASK)
                    if not current["activated"] or canonical_hash(task) != TASK_PIN:
                        return False
                    if work.get("panel"):
                        registry = yaml.safe_load(
                            (ROOT / "ops/arc_solve_registry.yaml").read_text()
                        )["games"]
                        if freeze(work["panel"]["installed_roster"], registry) != work["panel"]:
                            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def controls(value: Json, raw: Path) -> list[Json]:
    """Actual fresh children qualify positive, missing, deliberate-error and rehashed controls."""
    with bindings():
        return list(base.controls(value, raw))


def publish(value: Json, work: Json, output: Path, raw: Path) -> None:
    """The unchanged publisher and validators retain findings before exposing terminal bytes."""
    with bindings():
        base.publish(value, work, output, raw)


def run(output: Path, private: Path, *, private_e2e: bool = False) -> int:
    """Collect exactly the frozen panel after qualification; fixtures never supply live evidence."""
    started = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(started)
    raw.mkdir(parents=True, exist_ok=True)
    progress("preconditions_before")
    specs = [] if private_e2e else plan(private)
    atomic_json(
        raw / "command_manifest.json",
        dict(
            commands=specs,
            MODEL_SPECS=[],
            task_cap_s=4800,
            heartbeat_s=30,
            arm_order=["off", "on"],
            seeds=[11, 22],
            window=120,
        ),
    )
    checks, hashes, resources = preconditions(private, ROOT)
    refs = [
        snapshot(Path(name), raw / "inputs", digest) for name, digest in hashes.items() if digest
    ]
    panel, arcade = None, None
    registry = yaml.safe_load((ROOT / "ops/arc_solve_registry.yaml").read_text())["games"]
    if private_e2e:
        checks.append(
            gate("private_constructed_control", private, "research_observations", True, None)
        )
    else:
        try:
            progress("sdk_metadata_before")
            arcade = sdk()
            panel = freeze([str(g.game_id) for g in arcade.available_environments], registry)
            progress("sdk_metadata_after")
            for game in panel["missing_games"]:
                checks.append(
                    gate(
                        "installed_frozen_game", ROOT / "environment_files", game, "installed", None
                    )
                )
            atomic_json(raw / "frozen_panel.json", panel)
        except (ImportError, OSError, ValueError) as exc:
            checks.append(
                gate("sdk_access", ROOT / "environment_files", "frozen_sdk_games", True, str(exc))
            )
    progress("preconditions_after_validation_before")
    receipts = execute(specs, raw / "checks")
    rows = []
    if panel and all(r["passed"] for r in receipts) and resources.get("task_sha256") == TASK_PIN:
        progress("live_panel_before", 0, 8)
        for index, unit in enumerate(panel["units"]):
            if unit["game"] in panel["missing_games"]:
                progress("absent_game_slot_blocked", index + 1, 7 - index)
                continue
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
                deadline=180,
                scope="measurement",
            )
            terminal_path = episode_raw / "episode.json"
            terminal = json.loads(terminal_path.read_bytes()) if terminal_path.exists() else None
            rows.append(base.reduce_episode(unit, terminal, episode_raw))
            receipts.append(receipt)
            progress("live_panel_progress", index + 1, 7 - index)
        progress("live_panel_after", len(rows), 8 - len(rows))
    original = json.loads((ROOT / ORIGINAL).read_bytes())
    receipts.extend(original.get("global_repository_health", []))
    shards = {str(path): sha256_file(path) for path in raw.rglob("*") if path.is_file()}
    work = dict(
        panel=panel,
        rows=rows,
        failures=checks,
        source_artifact_hashes=hashes,
        source_input_receipts=refs,
        raw_shard_hashes=shards,
        code_config_hashes={name: sha256_file(ROOT / name) for name in OWNED},
        preconditions_checked=resources,
        historical_model_provenance=original["historical_model_provenance"],
        phase_spans=[
            dict(
                phase="qualification_and_live_panel",
                started_monotonic_ns=started,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
        duration_s=(time.monotonic_ns() - started) / 1e9,
        run_date="20261010",
    )
    value = build(work, receipts, raw, output)
    progress("cold_controls_before")
    receipts.extend(controls(value, raw))
    value = build(work, receipts, raw, output)
    progress("terminal_validation_before")
    publish(value, work, output, raw)
    progress("terminal_publication_after")
    return 0
