"""REQ-REPORT-8486: new qualification keeps original source controls and authority intact."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v728_source_control_panel as _legacy
from carnot.reporting import v730_contract_methods as _methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt

Json = dict[str, Any]
legacy, methods = _legacy, _methods
ROOT = methods.ROOT
NAME, TASK = (
    "experiment_8486_v730_source_control_qualification",
    "exp8486-source-control-qualification",
)
CLI, TEST = (
    f"scripts/experiments/{NAME}.py",
    "tests/python/test_v730_source_control_qualification_8486.py",
)
OWNED = [
    "python/carnot/reporting/v730_source_control_qualification.py",
    "python/carnot/reporting/v730_source_control_runner.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
METHODS_PIN = "sha256:552831473fa8df1fd272ae690082cbf23513e559fa9648e9ab0f588f603bb667"
ORIGINAL_PIN = "sha256:96602b67ce48980cfddaba1f94aa5a3c40d8aa554874885ce451859e114dbbc9"
ORIGINAL = ROOT / "results" / (legacy.NAME + ".json")
read, reference, checksum, sha256_file = (
    legacy.read,
    legacy.reference,
    legacy.checksum,
    legacy.sha256_file,
)
_MEASURE, _BUILD, _REPLAY = legacy.measure, legacy.build, legacy.replay
_METHODS = SimpleNamespace(
    **dict(
        vars(methods),
        scalar_dimension=legacy.methods.scalar_dimension,
        legacy=legacy.methods.legacy,
        CLI=CLI,
    )
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Visible actual counts let supervisors distinguish ongoing work from a stalled task."""
    print(f"[exp8486] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Only current task identity changes; the original producer checks remain independent."""
    with patch.multiple(
        legacy,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        methods=_METHODS,
        build=build,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path) -> Json:
    """Authenticate the required methods receipt before permitting shipped control freezing."""
    progress("authenticate_v730_methods", 0, 1)
    path = root / "results" / (methods.NAME + ".json")
    failures: list[Json] = []
    consumers = [
        reader_receipt(methods.TASK, path.parent, field=f)
        for f in ["methods_ready_score", "source_inputs_ready_score"]
    ]
    refs: list[Json] = []
    try:
        upstream = read(dict(path=str(path), sha256=METHODS_PIN))
        terminal = read(
            dict(
                path=upstream["terminal_validation_sidecar_path"],
                sha256=sha256_file(Path(upstream["terminal_validation_sidecar_path"])),
            )
        )
        publication = terminal["publication"]
        sidecar = read_bound_sidecar(path, Path(publication["sidecar_path"]))
        if not all(c["passed"] for c in consumers) or not sidecar["report"]["passed"]:
            raise ValueError("required_methods_consumers")
        refs = [reference(path), reference(Path(publication["sidecar_path"]))]
    except (OSError, ValueError, KeyError) as error:
        failures.append(
            dict(
                legacy.failure(
                    path,
                    "methods_and_source_inputs_ready_score",
                    1,
                    None if isinstance(error, FileNotFoundError) else str(error),
                ),
                upstream_id=methods.TASK,
                artifact_hash=METHODS_PIN,
            )
        )
    original_authority = _METHODS.authority

    def admitted(location: Path, scratch: Path) -> Json:
        """An absent required receipt cannot be bypassed by apparently usable cached rows."""
        authority = dict(original_authority(location, scratch))
        authority["gate_check_summary"] += failures
        return authority

    with binding(), patch.object(_METHODS, "authority", admitted):
        work = dict(_MEASURE(root, raw))
    work.update(methods_consumer=consumers[0], source_consumer=consumers[1])
    pin = producer_pin_control(raw / "producer_pin")
    atomic_json(raw / "producer_pin_control.json", pin)
    work["producer_pin_control_reference"] = reference(raw / "producer_pin_control.json")
    work["primitive_files"].append(work["producer_pin_control_reference"])
    if not all(
        pin[k]
        for k in ["valid_original_passed", "forged_producer_rejected", "local_hashes_consistent"]
    ):
        work["owned_failure"] = True
        work["failures"].append(
            dict(
                legacy.failure(raw, "independent_original_producer_pin_control", True, pin),
                upstream_id=TASK,
            )
        )
    work["source_refs"] += refs
    work["code_refs"] += [reference(ROOT / "python/carnot/reporting/v728_source_control_panel.py")]
    atomic_json(raw / "measurement.json", work)
    progress("controls_qualified_for_validation", len(work["panel"]["rows"]), 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """A new terminal receipt qualifies engineering controls without rewriting old failures."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(
        experiment_id=8486,
        task_id=TASK,
        milestone="2026.10.730",
        run_date="20261013",
        honest_verdict=f"complete_{value['verdict_class']}_v730_source_control_qualification",
        methods_consumer=work["methods_consumer"],
        source_consumer=work["source_consumer"],
        producer_pin_control_reference=work["producer_pin_control_reference"],
        superseded_qualification=dict(
            path=str(ORIGINAL),
            sha256=ORIGINAL_PIN,
            verdict="complete_disqualified_v728_source_control_panel",
            terminal_bytes_preserved=True,
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation") for n in [8487, 8488]
        ],
    )
    value["historical_authority_limits"]["V729"] = (
        "Truncated authority; ten tasks cascade-blocked; source science unmeasured."
    )
    value["historical_authority_limits"]["V728"] = (
        "Qualified journal equivalence; complete-service null speedup1.0447833367 CI95[0.9942346768,1.1095895448]."
    )
    for ref in value["cited_upstream_artifacts"]:
        if ref["sha256"] == METHODS_PIN:
            ref["imported_fields"] = [
                "methods_ready_score",
                "source_inputs_ready_score",
                "terminal_validation_sidecar_path",
            ]
    value["historical_model_provenance"][0]["original_producer_hashes"] = work["upstream_refs"]
    value["field_principles"].update(
        {
            k: "Bind current qualification to original independent producer seals; historical exposure and failures remain explicit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold shipped reconstruction checks frozen authority, source controls and local receipts."""
    with binding():
        return bool(_REPLAY(path))


@contextmanager
def original_binding(scratch: Path) -> Iterator[None]:
    """Historical assertions read sealed original authority rather than the current roadmap."""
    from carnot.reporting.historical_consumer_roots_8402 import inject

    old = read(dict(path=str(ORIGINAL), sha256=ORIGINAL_PIN))
    work = read(old["primitive_reference"])
    refs = []
    for ref in work["authority"]["authority_snapshots"].values():
        if ref.get("exists"):
            source = Path(ref["snapshot_path"])
            if sha256_file(source) != ref["sha256"]:
                raise ValueError("original_authority_pin")
            refs.append(dict(ref, path=str(source), snapshot_path=str(source)))
    with inject(dict(refs=refs), scratch):
        yield


def producer_pin_control(raw: Path) -> Json:
    """Change producer bytes and every local reference before testing the independent seal.

    Merely changing a reference hash fails at file authentication. This fixture
    reaches the later original-pin rejection because all local hashes agree.
    """
    progress("original_producer_pin_control", 0, 2)
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    old = read(dict(path=str(ORIGINAL), sha256=ORIGINAL_PIN))
    work = deepcopy(read(old["primitive_reference"]))
    for ref in work["primitive_files"]:
        target = raw / Path(ref["path"]).name
        target.write_bytes(Path(ref["path"]).read_bytes())
    work["primitive_files"] = [
        reference(raw / Path(ref["path"]).name) for ref in work["primitive_files"]
    ]
    receipt = legacy.child(
        "pin_fixture",
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            "-c",
            "print('independent original pin fixture',flush=True)",
        ],
        raw / "logs",
        deadline=30,
    )
    candidate = raw / (legacy.NAME + ".json")
    atomic_json(raw / "measurement.json", work)
    atomic_json(candidate, legacy.build(work, [receipt], raw, candidate))
    valid = legacy.replay(candidate)
    original = work["upstream_refs"][0]
    forged = deepcopy(read(original))
    forged["forged_pin_fixture"] = True
    forged["reproducibility_checksum"] = checksum(forged)
    target = raw / "forged_producer.json"
    atomic_json(target, forged)
    replacement = reference(target)
    for key in ["upstream_refs", "source_refs"]:
        work[key] = [replacement if ref == original else ref for ref in work[key]]
    work["history"]["references"][0] = replacement
    atomic_json(raw / "measurement.json", work)
    atomic_json(candidate, legacy.build(work, [receipt], raw, candidate))
    rejected = not legacy.replay(candidate)
    evidence = dict(
        valid_original_passed=valid,
        forged_producer_rejected=rejected,
        local_hashes_consistent=sha256_file(target) == replacement["sha256"],
        forged_producer_reference=replacement,
        forged_candidate_reference=reference(candidate),
        original_terminal_reference=dict(path=str(ORIGINAL), sha256=ORIGINAL_PIN),
        original_rejection_statement="python/carnot/reporting/v728_source_control_panel.py:554",
        receipt=receipt,
    )
    atomic_json(raw / "producer_pin_control.json", evidence)
    progress("original_producer_pin_control_finished", 2, 0)
    return evidence
