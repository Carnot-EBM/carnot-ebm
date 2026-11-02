"""REQ-VERIFY-8656: source roles prevent consumer coverage from blessing producers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coverage import CoverageData
from coverage.exceptions import DataError

from carnot.reporting import streaming_custody_8642 as c
from carnot.reporting import streaming_source_8629 as old
from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]
ROOT = old.ROOT
HISTORICAL = {
    "results/experiment_8629_v741_streaming_source_prototype.json": "sha256:9ea7bc5448d8f8c84b9c9e9ee5ff1d7e70892f51468827c054e10e7767b3435b",
    "results/experiment_8642_v742_streaming_execution_custody.json": "sha256:0c8f23991c6986a987d5de6df8d664ba544b2db2a486b2c9cd25ba0c4b0c9169",
}
ORIGINAL_DRIVER = "sha256:dcd6388a5d0a39f403a05b180abee01c20b6f6407df81a90217d54b3e1098378"
FROZEN = "results/raw/experiment_8629_v741_streaming_source_prototype/attempts/initial_disqualified/frozen_code"
FAILED = ["test_measure_build_and_cold_replay", "test_private_cli_modes_and_publication"]
ROLES = {"historical_driver", "streaming_producer", "current_adapter", "historical_consumer"}


def gate(path: Path, field: str) -> Json:
    """Keep missing files, absent fields and a recorded zero distinct for downstream readers."""
    value = json.loads(path.read_bytes()) if path.is_file() else {}
    observed = value.get(field)
    reason = (
        "missing_file"
        if not path.is_file()
        else "missing_field"
        if field not in value
        else "observed_zero"
        if observed == 0
        else None
    )
    return dict(
        upstream=path.stem,
        check=field,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=1,
        observed=observed,
        missing_reason=reason,
        passed=type(observed) is int and observed == 1,
    )


def authority(root: Path, raw: Path) -> Json:
    """Pin original failures independently; new measurements cannot rewrite historical authority."""
    result: Json = dict(historical=[], gates=[], views={}, failed_owned_test_identities=[])
    for name, expected in HISTORICAL.items():
        path = root / name
        observed = sha256_file(path) if path.is_file() else None
        if observed != expected:
            result["gates"].append(
                dict(
                    upstream=path.stem,
                    check="historical_identity",
                    path=str(path),
                    sha256=observed,
                    field="sha256",
                    operator="==",
                    expected=expected,
                    observed=observed,
                    missing_reason="missing_file" if observed is None else "hash_mismatch",
                )
            )
            return result
        result["historical"].append(dict(origin_path=str(path), reference=c.freeze(path, raw)))
    prior = json.loads((root / next(reversed(HISTORICAL))).read_bytes())
    ref = prior["measurement_reference"]
    if not c.authenticate(ref):
        raise ValueError("historical_measurement")
    work = json.loads(Path(ref["path"]).read_bytes())
    result["measurement_reference"] = c.freeze(Path(ref["path"]), raw)
    original = []
    first = json.loads((root / next(iter(HISTORICAL))).read_bytes())
    original_pins = {r["path"]: r["sha256"] for r in first["code_config_hashes"]}
    original_pins[str(ROOT / old.TEST)] = (
        "sha256:d0d61fe18394ee0a11b6e6195928c41d019dc87afda0be9bd15b4d0cb98d93a7"
    )
    for name in [*old.OWNED, old.TEST]:
        path = root / FROZEN / name
        if sha256_file(path) != original_pins[str(ROOT / name)]:
            raise ValueError("historical_source_pin")
        original.append(dict(origin_path=str(ROOT / name), reference=c.freeze(path, raw)))
    if original[2]["reference"]["sha256"] != ORIGINAL_DRIVER:
        raise ValueError("historical_driver")
    result["views"]["8629"] = original
    if not all(c.authenticate(r["reference"]) for r in work["original_authority"]):
        raise ValueError("historical_closure")
    original64 = [
        dict(r, reference=c.freeze(Path(r["reference"]["path"]), raw))
        for r in work["original_authority"]
        if "8642" in r["origin_path"]
    ]
    result["views"]["8642"] = original64 + original
    result["historical_driver_reference"] = original[2]["reference"]
    result["driver_mismatch"] = dict(
        path=str(ROOT / old.OWNED[2]),
        expected=ORIGINAL_DRIVER,
        observed=sha256_file(ROOT / old.OWNED[2]),
        reproduced=sha256_file(ROOT / old.OWNED[2]) != ORIGINAL_DRIVER,
    )
    result["failed_owned_test_identities"] = [
        "tests/python/test_streaming_custody_8642.py::" + n for n in FAILED
    ]
    failed = next(
        m for m in prior["child_shard_manifest"] if m["name"].startswith("test_streaming_custody")
    )
    result["failure_logs"] = [
        c.freeze(Path(failed["receipt"][k]), raw) for k in ("stdout_path", "stderr_path")
    ]
    result["consumer_failures"] = [
        dict(
            name=m["name"],
            exit_code=m["receipt"]["exit_code"],
            error=m["receipt"].get("capture_error"),
            logs=[c.freeze(Path(m["receipt"][k]), raw) for k in ("stdout_path", "stderr_path")],
        )
        for m in prior["child_shard_manifest"]
        if m["receipt"].get("capture_error")
    ]
    return result


def private_command(view: list[Json], argv: list[str], scratch: Path) -> list[str]:
    """Mount exact original bytes at their recorded paths without changing any shared reader."""
    command = [
        "bwrap",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        str(scratch),
        str(scratch),
        "--bind",
        "/tmp",
        "/tmp",
        "--bind",
        "/var/tmp",
        "/var/tmp",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
    ]
    for row in view:
        if not c.authenticate(row["reference"]):
            raise ValueError("historical_source")
        command += ["--ro-bind", row["reference"]["path"], row["origin_path"]]
    return command + ["--", *argv]


def role_inventory(suite: str, role: str, sources: list[Path], raw: Path) -> Json:
    """Seal sources before execution so a later manifest cannot relabel changed bytes."""
    if role not in ROLES:
        raise ValueError("source_role")
    return dict(
        suite=suite,
        role=role,
        sources=[
            dict(source=str(p.absolute()), source_sha256=sha256_file(p), reference=c.freeze(p, raw))
            for p in sources
        ],
    )


def capture(base: Path, raw: Path, role: Json) -> list[Json]:
    """Read real SQLite immediately, requiring statements and authenticated role sources."""
    sources = {r["source"]: r for r in role["sources"]}
    for row in sources.values():
        if not c.authenticate(row["reference"]) or (
            role["role"] != "historical_driver"
            and sha256_file(Path(row["source"])) != row["source_sha256"]
        ):
            raise ValueError("source_mismatch")
    rows = []
    for path in sorted(p for p in base.parent.glob(base.name + "*") if p.is_file()):
        data = CoverageData(basename=str(path))
        data.read()
        if set(data.measured_files()) - sources.keys():
            raise ValueError("foreign_source_data")
        inventory = [
            dict(
                source=f,
                source_sha256=sources[f]["source_sha256"],
                lines=sorted(data.lines(f) or []),
                contexts={str(n): sorted(v) for n, v in data.contexts_by_lineno(f).items()},
            )
            for f in sorted(data.measured_files())
        ]
        rows.append(
            dict(
                reference=c.freeze(path, raw),
                original_path=str(path),
                unsuffixed=path == base,
                inventory=inventory,
            )
        )
    if not any(r["lines"] for shard in rows for r in shard["inventory"]):
        raise ValueError("missing_child_statements")
    return rows


def combine(manifests: list[Json], role: str, sources: list[Path], scratch: Path) -> Json:
    """Only current data for this exact role can supply its statement gate."""
    if not manifests or any(m["role"]["role"] != role for m in manifests):
        raise ValueError("source_role")
    return c.combine([r for m in manifests for r in m["shards"]], sources, scratch)
