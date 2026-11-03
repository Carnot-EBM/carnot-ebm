"""REQ-VERIFY-8771: bind historical readers without changing their assertions."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Any
import xml.etree.ElementTree as ET

import yaml

from carnot.reporting import compiled_sparse_study_8719 as compiled
from carnot.reporting import compiled_sparse_runner_8719 as runner
from carnot.reporting import v747_contract_methods as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = compiled.ROOT
PRESERVED = "openspec/change-proposals/research-roadmap-v747-preserved-20261102.md"
DESIGN_PIN = "sha256:4a62f587ca9d03a724945ddbb6e8f4d69995128dec7a015ce4bf738383c82109"
OLD = "results/experiment_8758_v750_lifecycle_event_qualification.json"
OLD_PIN = "sha256:8e0c904ab7ff9859d002293e05fc2aa3957ca0c0b8379019afe125d448431ac3"
HISTORICAL_CLI = "scripts/experiments/experiment_8719_v747_compiled_sparse_update.py"
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]


def fixture(root: Path) -> tuple[Path, Json]:
    """Reuse the archive-root layout with private authority files and copied dependencies.

    Copies prevent pytest cleanup from resolving a link into production evidence.
    The manifest checks authority, source and artifact bytes before each use.
    """
    compiled.verify(dict(path=str(ROOT / PRESERVED), sha256=DESIGN_PIN))
    root.mkdir(parents=True)
    native = compiled.read(
        dict(
            path=str(ROOT / "results/experiment_8719_v747_compiled_sparse_update.json"),
            sha256="sha256:9d27b1f8ca99b50d0452d7e1d204c18223a16b7eec36208d8bc5c44830c6f1ac",
        )
    )
    native_work = compiled.read(native["primitive_reference"])
    closure = Path(native_work["raw"]) / "source_closure"
    for source in closure.rglob("*"):
        if source.is_file():
            target = root / source.relative_to(closure)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    for relative in [compiled.PRODUCER, compiled.previous.PROTOCOL]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    text = (ROOT / PRESERVED).read_text()
    tasks = parse_design(text, milestone=contract.MILESTONE)[1]
    if canonical_hash(tasks) != "sha256:" + contract.PIN:
        raise ValueError("original_v747_digest")
    (root / contract.DESIGN).write_text(text)
    for name in [contract.ACTIVE, contract.STAGED]:
        (root / name).write_text(yaml.safe_dump(dict(milestone=contract.MILESTONE, tasks=tasks)))
    refs = [
        compiled.reference(root / p) for p in [contract.DESIGN, contract.ACTIVE, contract.STAGED]
    ]
    refs += [compiled.reference(ROOT / p) for p in compiled.OWNED + compiled.TESTS]
    # Original artifacts and source pins remain inputs; none is republished here.
    refs += compiled.inputs()["refs"]
    refs += [compiled.reference(root / p) for p in [compiled.PRODUCER, compiled.previous.PROTOCOL]]
    refs += [
        compiled.reference(p) for p in root.rglob("*") if p.is_file() and p != root / HISTORICAL_CLI
    ]
    manifest = root / "authority_manifest.json"
    atomic_json(manifest, dict(references=refs, canonical_tasks_sha256=contract.PIN))
    ref = compiled.reference(manifest)
    wrapper = root / HISTORICAL_CLI
    wrapper.write_text(
        "from pathlib import Path\nimport sys\n"
        f"sys.path[:0] = {[str(ROOT / 'python'), str(ROOT)]!r}\n"
        "from carnot.reporting.lifecycle_archive_authority_8771 import historical_main\n"
        f"raise SystemExit(historical_main(Path({str(root)!r}), {ref['sha256']!r}))\n"
    )
    for path in [
        manifest,
        wrapper,
        *(root / p for p in [contract.DESIGN, contract.ACTIVE, contract.STAGED]),
    ]:
        path.chmod(0o444)
    verify_fixture(root, ref["sha256"])
    return root, ref


def verify_fixture(root: Path, digest: str) -> Json:
    """An independently supplied manifest hash prevents a local edit from blessing itself."""
    value = compiled.read(dict(path=str(root / "authority_manifest.json"), sha256=digest))
    for ref in value["references"]:
        compiled.verify(ref)
    authority = contract.authority(root)
    if value["canonical_tasks_sha256"] != contract.PIN or not authority["ready"]:
        raise ValueError("immutable_v747_authority")
    return dict(
        authority, canonical_tasks_sha256=contract.PIN, authority_root=str(root), passed=True
    )


def bind(root: Path, digest: str) -> None:
    """Only the historical module globals change inside a private child interpreter."""
    verify_fixture(root, digest)
    compiled.ROOT, compiled.CLI = root, str(root / HISTORICAL_CLI)


def pytest_configure(config: Any) -> None:
    """An explicit plugin binding leaves ordinary tests and production readers unchanged."""
    root = os.environ.get("CARNOT_8771_ARCHIVE")
    if root:
        bind(Path(root), os.environ["CARNOT_8771_ARCHIVE_SHA256"])


def historical_main(root: Path, digest: str) -> int:
    """The real historical CLI uses the same fixture and validators as its pytest parent."""
    bind(root, digest)
    return int(runner.main())


def reproduce(raw: Path) -> tuple[Json, list[Json], list[str]]:
    """Run the retained eight-node command with current clocks and private output paths."""
    old_ref = dict(path=str(ROOT / OLD), sha256=OLD_PIN)
    old = compiled.read(old_ref)
    retained = next(r for r in old["validation_receipts"] if r["name"] == "tests_2")
    for stream in ["stdout", "stderr"]:
        compiled.verify(dict(path=retained[stream + "_path"], sha256=retained[stream + "_sha256"]))
    nodes = [s for s in retained["argv"] if s.startswith("tests/") and "::" in s]
    xml = raw / "reproduced.xml"
    receipt = child(
        "original_tests_2_failure",
        [
            sys.executable,
            "-m",
            "pytest",
            *COMMON,
            "-q",
            *nodes,
            "--junitxml=" + str(xml),
            "--basetemp=" + str(raw / "pytest"),
        ],
        raw,
        expected=1,
        deadline=120,
    )
    failed = [
        n.attrib["name"] for n in ET.parse(xml).iter("testcase") if n.find("failure") is not None
    ]
    if not receipt["passed"] or failed != ["test_authority"]:
        raise ValueError("historical_failure_not_reproduced")
    receipt.update(
        failed_nodes=failed, retained_receipt=retained, old_primary_remains_disqualified=True
    )
    refs = [old_ref, compiled.reference(xml)]
    refs += [compiled.reference(Path(receipt[k])) for k in ["stdout_path", "stderr_path"]]
    return receipt, refs, nodes
