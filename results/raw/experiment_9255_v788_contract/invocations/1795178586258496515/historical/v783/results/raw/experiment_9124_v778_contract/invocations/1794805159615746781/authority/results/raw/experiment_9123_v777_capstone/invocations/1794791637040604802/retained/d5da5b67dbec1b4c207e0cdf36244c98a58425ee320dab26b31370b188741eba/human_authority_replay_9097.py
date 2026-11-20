"""REQ-REPORT-9097-V776: replay original mapping in an isolated interpreter.

This worker imports no current Carnot module. Its only project imports come
from the authenticated snapshot directory, which is checked before import.
"""

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def checked(reference: dict[str, Any]) -> bytes:
    """Authenticate operands before their fields can affect historical replay."""
    data = Path(reference["path"]).read_bytes()
    if "sha256:" + hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("worker_operand_hash")
    return data


def main(job_path: Path) -> None:
    """Run the original validator and retain its whole-question role projection."""
    job = json.loads(job_path.read_bytes())
    for reference in job["modules"].values():
        checked(reference)
    for name in ["carnot", "carnot/reporting"]:
        if (Path(job["snapshot_root"]) / name / "__init__.py").read_bytes() != b"":
            raise ValueError("worker_package_closure")
    sys.path.insert(0, job["snapshot_root"])
    from carnot.reporting import human_label_custody_8389 as historical

    manifest = json.loads(checked(job["manifest"]))
    result = historical.validate_manifest(manifest)
    roles = json.loads(checked(job["roles"]))["partition"]["roles"]
    role_index = {
        group: {"fit": "TRAIN", "tune": "CAL", "test": "TEST"}[role]
        for role, groups in roles.items()
        for group in groups
    }
    rows = [
        dict(
            unit_id=r["unit_id"],
            qid=r["qid"],
            group_id=r["cluster"],
            annotator1=r["human_primary"],
            annotator2=r["human_sensitivity"],
            role=role_index[r["cluster"]],
        )
        for r in result["rows"]
    ]
    imported = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("carnot.reporting."):
            path = Path(module.__file__).resolve()
            reference = dict(
                path=str(path), sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            )
            key = name.rsplit(".", 1)[1]
            if job["modules"].get(key) != reference:
                raise ValueError("worker_import_closure")
            imported[key] = reference
    target = job_path.parent / "historical_projection.json"
    value = dict(rows=rows, imported_modules=imported)
    if target.exists():
        if json.loads(target.read_bytes()) != value:
            raise ValueError("worker_projection_drift")
    else:
        target.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
        target.chmod(0o400)
    print("[exp9097] phase=historical_replay_complete rows=" + str(len(rows)), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
