"""REQ-VERIFY-8587: execute the real publication child before coverage sealing."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory
from typing import Iterator
from unittest.mock import patch

from carnot.reporting import arc_publication_8587 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.historical_consumer_roots_8402 import inject

Json = e.Json
VALIDATION = e.runner.validation
HISTORICAL = [*e.prior.HISTORICAL, e.prior.TEST]
AUTHORITY = (
    e.old.ROOT
    / "results/raw/experiment_8552_v735_contract_methods/frozen/c5841348e035db7600bb48949a1a1740c31447f353fa1a01af31e7ce851da868.yaml"
)
AUTHORITY_PIN = "sha256:c5841348e035db7600bb48949a1a1740c31447f353fa1a01af31e7ce851da868"


def publication_cases(target: Path) -> Json:
    """Real validators reject altered bytes while prior checked bytes stay intact."""
    started = time.monotonic()
    directory = target.parent / "private_publication"
    output = directory / (e.NAME + ".json")
    work = e.runner.empty_work(directory / "scratch")
    work["duration_s"] = time.monotonic() - started
    path = directory / "blocked_work.json"
    atomic_json(path, work)
    value = e.build(path, output)
    valid = e.prior.publish(value, output, directory / "valid")
    failed = deepcopy(work)
    failed["checks"].append(dict(name="owned_failure", passed=False))
    failed_path = directory / "failed_work.json"
    atomic_json(failed_path, failed)
    owned = e.prior.publish(e.build(failed_path, output), output, directory / "failed")
    digest = sha256_file(output)
    altered = deepcopy(value)
    altered["completed_count"] = 999
    altered["reproducibility_checksum"] = e.old.canonical_hash(altered)
    rejected = False
    try:
        e.prior.publish(altered, output, directory / "invalid")
    except ValueError as exc:
        rejected = str(exc) == "candidate_rejected"
    candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
    consumer = e.panel_consumer(candidate)
    result = dict(
        passed=valid["passed"]
        and owned["passed"]
        and rejected
        and sha256_file(output) == digest
        and consumer["matched_expectation"],
        valid=valid,
        owned_failure=owned,
        invalid=dict(
            rejected=rejected,
            primary_unchanged=sha256_file(output) == digest,
            panel_consumer=consumer,
            candidate_sha256=sha256_file(candidate),
        ),
        immutable_snapshots=[dict(path=str(p), sha256=sha256_file(p)) for p in (path, failed_path)],
    )
    atomic_json(target, result)
    return result


def measurement(work: Json, root: Path, raw: Path, private: Path) -> None:
    """The diagnosed closure replaces the obsolete conditional reproduction."""
    try:
        history = e.historical_receipt()
        primary, original = e.old.authenticate(root)
    except (OSError, ValueError, KeyError) as exc:
        work["failures"][0].update(check="original_input_authentication", observed=str(exc))
        return
    work.update(
        publication_history=history,
        history=e.old.history_receipts(primary),
        episode_hashes=e.old.episode_hashes(original),
        sources={str(root / e.old.ORIGINAL): e.old.PIN, str(e.UPSTREAM): e.UPSTREAM_PIN},
        historical_model_provenance=primary["historical_model_provenance"],
    )
    work["checks"].append(dict(name="failure_reproductions", passed=history["passed"]))
    candidate = e.old.private_history(primary, original, private / "stored")
    receipt = e.runner.invoke(
        "original_terminal_replay",
        [sys.executable, "-u", str(root / e.old.panel.CLI), "--cold-replay", str(candidate)],
        raw / "measurement",
        deadline=240,
    )
    work["checks"].append(receipt)
    work["terminal_replays"].append(receipt)
    for name, flag, filename in [
        ("primitive_replay", "--primitives", "replay_rows.json"),
        ("original_fixtures", "--fixtures", "fixtures.json"),
    ]:
        target = raw / filename
        receipt = e.runner.invoke(
            name,
            [sys.executable, "-u", str(root / e.CLI), flag, str(target)],
            raw / "measurement",
            deadline=240,
        )
        work["checks"].append(receipt)
        if name == "primitive_replay":
            work["terminal_replays"].append(receipt)
        if receipt["passed"]:
            work["replay_rows" if name == "primitive_replay" else "fixtures"] = json.loads(
                target.read_bytes()
            )
    work["failures"] = []


def validation(work: Json, raw: Path, private: Path, config: Path) -> None:
    """Run original assertions, then a bounded private publication before sealing."""
    VALIDATION(work, raw, private, config)
    target = raw / "publication_cases.json"
    receipt = e.runner.invoke(
        "publication_closure",
        [sys.executable, "-u", str(e.old.ROOT / e.CLI), "--publication-cases", str(target)],
        raw / "validation",
        deadline=240,
    )
    work["checks"].append(receipt)
    if receipt["passed"]:
        work["publication_cases"] = json.loads(target.read_bytes())
        work["publication_cases_reference"] = dict(path=str(target), sha256=sha256_file(target))
        work["checks"].append(
            dict(name="publication_cases", passed=work["publication_cases"]["passed"])
        )


@contextmanager
def historical_authority() -> Iterator[None]:
    """Carry sealed historical reads into genuine CLI children without editing authority."""
    import subprocess
    from typing import Any

    e.old.panel.checked_bytes(str(AUTHORITY), AUTHORITY_PIN)
    reference = dict(
        path=str(AUTHORITY),
        snapshot_path=str(AUTHORITY),
        source_path=str(e.old.ROOT / "research-roadmap.yaml"),
        sha256=AUTHORITY_PIN,
        exists=True,
    )
    original = subprocess.Popen

    def popen(args: Any, *a: Any, **kwargs: Any) -> Any:
        argv = list(args)
        legacy = str(e.old.ROOT / e.prior.CLI)
        if legacy in argv:
            index = argv.index(legacy)
            argv[index : index + 1] = [str(e.old.ROOT / e.CLI), "--legacy"]
        return original(argv, *a, **kwargs)

    with TemporaryDirectory(prefix="exp8587-authority-", dir="/var/tmp") as directory:
        authority = inject(dict(refs=[reference]), Path(directory))
        children = patch.object(subprocess, "Popen", popen)
        with authority, children:
            yield


@contextmanager
def adapters() -> Iterator[None]:
    """Keep current identity separate from the unchanged original test workers."""
    identity = patch.multiple(
        e.old,
        NAME=e.NAME,
        TASK=e.TASK,
        TASK_PIN=e.TASK_PIN,
        CLI=e.CLI,
        TEST=e.TEST,
        OWNED=e.OWNED,
        RUN_DATE="20261016",
        build=e.build,
        replay=e.replay,
        progress=e.progress,
        REQUIRED=e.old.REQUIRED | {"publication_closure", "publication_cases"},
    )
    execution = patch.multiple(
        e.runner,
        measurement=measurement,
        validation=validation,
        HISTORICAL=HISTORICAL,
        publish=e.prior.publish,
    )
    with identity, execution, patch.object(e.prior, "panel_consumer", e.panel_consumer):
        yield


def main(argv: list[str] | None = None) -> int:
    """Use bounded shipped orchestration; private fixtures never call a controller."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    if "--legacy" in arguments:
        arguments.remove("--legacy")
        with historical_authority():
            return e.prior.main(arguments)
    if "--suite" in arguments:
        with patch.object(e.runner, "HISTORICAL", HISTORICAL):
            if arguments[arguments.index("--suite") + 1] == e.prior.TEST:
                with historical_authority():
                    return e.runner.main(arguments)
            return e.runner.main(arguments)
    with adapters():
        if "--publication-cases" in arguments:
            return int(
                not publication_cases(Path(arguments[arguments.index("--publication-cases") + 1]))[
                    "passed"
                ]
            )
        return e.runner.main(arguments)
