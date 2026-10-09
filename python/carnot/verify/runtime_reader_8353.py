"""REQ-VERIFY-8353: preserved authority cannot be inferred from a mutable plan."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import yaml  # type: ignore[import-untyped]

from carnot.reporting import v718_contract_replay as contract
from carnot.reporting import v718_replay_history as history
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v710_contract_replay import snapshot, require_reference
from carnot.verify import runtime_reader_8340 as legacy
from carnot.verify import runtime_reader_report_8340 as report

Json = dict[str, Any]
ROOT, UPSTREAM, old = legacy.ROOT, legacy.UPSTREAM, legacy.old
NAME = "experiment_8353_v720_runtime_reader_qualification"
TASK, MILESTONE = "exp8353-runtime-reader-qualification", "2026.10.720"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_runtime_reader_8353.py"
OWNED = [
    "python/carnot/verify/runtime_reader_8353.py",
    "python/carnot/verify/runtime_reader_execution_8353.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
checksum, parse_design, changes, reduce = (
    legacy.checksum,
    legacy.parse_design,
    legacy.changes,
    legacy.reduce,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush each real boundary so supervisors can distinguish work from a stall."""
    print(f"[exp8353] phase={phase} completed={completed} pending={pending}", flush=True)


def design(root: Path, milestone: str) -> Path:
    """Historical tasks select their preserved design rather than future prompts."""
    return (
        root
        / {
            "2026.10.716": history.OLD_DESIGN,
            "2026.10.717": history.PRIOR_DESIGN,
            MILESTONE: "openspec/change-proposals/research-roadmap-vNEXT.md",
        }[milestone]
    )


def private_authority(root: Path, milestone: str) -> Path:
    """Copy complete authority for controls without changing the active roadmap."""
    path = design(root, milestone)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(design(ROOT, milestone).read_bytes())
    tasks = parse_design(path.read_text(), milestone=milestone)[1]
    (root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(dict(milestone=milestone, tasks=tasks))
    )
    return root


def authority(root: Path, raw: Path, milestone: str = MILESTONE) -> Json:
    """The existing full-task reader compares explicit versioned design bytes."""
    try:
        with patch.object(history, "design", design):
            return dict(contract.authority(root, raw, milestone))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            activated=False,
            refs=[],
            gate_check_summary=[
                old.gate(root / "research-roadmap.yaml", "execution_authority", True, str(error))
            ],
        )


@contextmanager
def bindings() -> Iterator[None]:
    """Small version adapters preserve the qualified reader and its thresholds."""
    with ExitStack() as stack:
        for name, value in dict(
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            EXPERIMENT_ID=8353,
            design=design,
            authority=authority,
            progress=progress,
        ).items():
            stack.enter_context(patch.object(legacy, name, value))
        yield


def closure(root: Path, raw: Path) -> list[Json]:
    """Retain the transitive source bytes so historical custody survives later edits."""
    refs: list[Json] = []
    seen: set[str] = set()

    def retain(path: Path, expected: str | None = None) -> None:
        key = str(path)
        if key in seen:
            return
        if not path.is_file():
            if expected is not None:
                raise FileNotFoundError(key)
            return
        seen.add(key)
        ref = snapshot(path, raw / "historical_closure", "source")
        if expected is not None:
            require_reference(dict(ref, sha256=expected))
        refs.append(ref)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if (
                value.get("sha256")
                and (value.get("path") or value.get("snapshot_path"))
                and not value.get("op")
            ):
                retain(Path(value.get("snapshot_path") or value["path"]), value["sha256"])
            for key, item in value.items():
                if key in {
                    "code_config_hashes",
                    "raw_shard_hashes",
                    "historical",
                    "prior_validation_attempts",
                    "cited_upstream_artifacts",
                    "field_principles",
                }:
                    continue
                if key == "terminal_validation_sidecar_path" and isinstance(item, str):
                    retain(Path(item))
                if key.endswith("_path") and value.get(key[:-5] + "_sha256"):
                    retain(Path(item), value[key[:-5] + "_sha256"])
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for path in [
        UPSTREAM,
        "results/experiment_8340_v719_runtime_reader_qualification.json",
        "results/experiment_8326_runtime_reader_qualification.json",
    ]:
        primary = root / path
        retain(primary)
        if primary.is_file():
            value = json.loads(primary.read_bytes())
            walk(value)
            primitive = value.get("primitive_reference", {}).get("path")
            if primitive and Path(primitive).is_file():
                walk(json.loads(Path(primitive).read_bytes()))
            sidecar = (
                primary.parent
                / "raw"
                / primary.stem
                / "validators"
                / (sha256_file(primary)[7:] + ".json")
            )
            retain(sidecar)
            if sidecar.is_file():
                if not old.read_bound_sidecar(primary, sidecar)["report"]["passed"]:
                    raise ValueError("historical_terminal_rejected:" + str(primary))
                walk(json.loads(sidecar.read_bytes()))
    progress("historical_closure_retained", len(refs), 0)
    return refs


def measure(root: Path, raw: Path, *, reader_checks_passed: bool = True) -> Json:
    """Authenticate historical execution before inspecting actual current identities."""
    progress("historical_closure_before")
    failures = []
    try:
        retained = closure(root, raw)
    except (OSError, ValueError, KeyError, TypeError) as error:
        retained = []
        failures.append(old.gate(root / UPSTREAM, "historical_source_closure", True, str(error)))
    with bindings():
        work = legacy.measure(root, raw, authorize_probe=reader_checks_passed and not failures)
    work["checks"].extend(failures)
    for check in work["checks"]:
        if check.get("field") == "full_V719_authority":
            check["field"] = "full_V720_authority"
    work["historical_fixture_manifest"] = retained + [
        snapshot(design(ROOT, milestone), raw / "historical_authority", milestone)
        for milestone in ("2026.10.716", "2026.10.717")
    ]
    work["refs"].extend(work["historical_fixture_manifest"])
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Expose all owned receipts while keeping health separate from reader correctness."""
    with bindings():
        value = report.build(work, raw, receipts)
    value.update(
        random_seed=7208353,
        historical_fixture_manifest=work["historical_fixture_manifest"],
        runtime_identity=work["current"],
        lease_receipt=work["diagnostic"].get("lease_receipt", {}),
        primitive_rows=value["rows"],
    )
    value["field_principles"].update(
        {
            key: "Bind retained historical bytes and actual current runtime observations."
            for key in (
                "historical_fixture_manifest",
                "runtime_identity",
                "lease_receipt",
                "primitive_rows",
            )
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return dict(value)


def replay(path: Path) -> bool:
    """Fresh reduction authenticates the historical closure and every added field."""
    try:
        with bindings():
            if not report.replay(path):
                return False
        value = json.loads(path.read_bytes())
        work = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
        for ref in work["historical_fixture_manifest"]:
            require_reference(ref)
        return all(
            value[k] == expected
            for k, expected in dict(
                historical_fixture_manifest=work["historical_fixture_manifest"],
                runtime_identity=work["current"],
                lease_receipt=work["diagnostic"].get("lease_receipt", {}),
                primitive_rows=reduce(work, value["required_checks_passed"])["rows"],
            ).items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
