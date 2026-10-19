"""REQ-VERIFY-8358: preserve exact closures without importing historical code.

Producer seals define membership. Current readers may use frozen copies only
after their own code matches the producer seal; missing bytes remain evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch
import sys
from types import ModuleType, SimpleNamespace

from carnot.reporting import gatemate_history_8357 as history
from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
PINS = {
    8342: "sha256:7a3c431bec0253f3328cf1bc9c2ed469fb7338f3b367f4e3e2ffe7bc38b85171",
    8344: "sha256:48eb3ad7bccb2d869d561c3e021b50348fed3339bddbce2d493442ce7108f73b",
    8355: "sha256:a93696d2e2b7cf9ee7a676652b89f02f8d93872cf346de9fccd2bf884586f394",
    8357: "sha256:fe3e29d775966a50bef90040dbb64a0a492ef913d76ebf32389316af3ae35071",
}
TERMINAL_PINS = {
    8342: (
        "sha256:af7ce5f0b9381e3215ac4ba819221d755f0e1295ea35b2cf8488b2a650e905ff",
        "sha256:04d9bfe90c6c943c2dca4b9dfc36c478e6887b6477712faa9a11c95c9c8be666",
    ),
    8344: (
        "sha256:0aaf24b1780d6580d9e71efccbaf3e9f3142bec140e183597dc0d4f42bb72e56",
        "sha256:5cbaee62cce2150a755e3ac78684389aef51d49ef0dbf1d59149455723ab077b",
    ),
    8355: (
        "sha256:d827be8587a5d4d1062ec7cc9b68d2e85c787a6100e62c1b6f1e8beef03bf200",
        "sha256:700950f60765806e297cb0712a58582de5ed50736298f4e8e7eaba8dc6178286",
    ),
    8357: (
        "sha256:dbc8723970fad720debe612489f414194c215cc0a4c8df226546f1ccc51e4fd4",
        "sha256:45c165a82e5bfa39a9f42701ba216f42bef4309a10bd8ad04eee611dafc5dfc4",
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual boundaries so an unavailable operand cannot look like a stall."""
    print(f"[exp8358] phase={phase} completed={completed} pending={pending}", flush=True)


def requirements(value: Json) -> list[Json]:
    """Normalize existing producer schemas without changing their declared seals."""
    refs = []
    for key in ["source_artifact_hashes", "code_config_hashes", "raw_shard_hashes"]:
        source = value.get(key, [])
        if isinstance(source, dict):
            source = [dict(path=str(ROOT / p), sha256=h) for p, h in source.items()]
        refs.extend(r for r in source if r.get("sha256"))
    for key in ["work_reference", "replay_input_reference", "primitive_reference"]:
        if value.get(key):
            refs.append(value[key])
    return list(
        {(r["path"], r["sha256"]): dict(path=r["path"], sha256=r["sha256"]) for r in refs}.values()
    )


def capture(path: Path, pin: str, raw: Path, index: dict[str, list[Path]]) -> Json:
    """Authenticate terminal custody before copying compact source references.

    Recovery reuses the existing content-addressed reader. The supplied index
    contains captured producer files; this adapter never executes recovered code.
    """
    path = path.absolute()
    if sha256_file(path) != pin:
        raise ValueError("primary_seal_drift")
    value = json.loads(path.read_bytes())
    identity = value["experiment_id"]
    if identity in PINS and pin != PINS[identity]:
        raise ValueError("producer_anchor_drift")
    side = path.parent / "raw" / path.stem / "validators" / (pin[7:] + ".json")
    bound = read_bound_sidecar(path, side)
    terminal_path = Path(value["terminal_validation_sidecar_path"])
    terminal = json.loads(terminal_path.read_bytes())
    if (
        identity in TERMINAL_PINS
        and (sha256_file(side), sha256_file(terminal_path)) != TERMINAL_PINS[identity]
    ):
        raise ValueError("terminal_seal_drift")
    if (
        bound["report"]["passed"] is not True
        or terminal["publication"]["primary_sha256"] != pin
        or terminal["publication"]["primary_path"] != str(path)
    ):
        raise ValueError("terminal_binding_drift")
    refs = [reference(path), reference(side), reference(terminal_path), *requirements(value)]
    rows = []
    for i, ref in enumerate(refs):
        progress("closure", i, len(refs) - i)
        rows.append(history.recover(ref, raw, raw / str(i), index, []))
    return dict(primary_sha256=pin, rows=rows, original_path=str(path), experiment_id=identity)


def verify(bundle: Json) -> Json:
    """Recompute membership from sealed primary bytes so rehashing cannot grant trust."""
    history.verify(bundle["rows"])
    primary, side, terminal = [
        json.loads(checked(r["reference"]).read_bytes()) for r in bundle["rows"][:3]
    ]
    pin = bundle["primary_sha256"]
    if (
        bundle["experiment_id"] != primary["experiment_id"]
        or bundle["original_path"] != bundle["rows"][0]["original_path"]
    ):
        raise ValueError("closure_identity_drift")
    if (
        primary["experiment_id"] in TERMINAL_PINS
        and tuple(r["expected_sha256"] for r in bundle["rows"][1:3])
        != TERMINAL_PINS[primary["experiment_id"]]
    ):
        raise ValueError("terminal_seal_drift")
    if bundle["rows"][0]["expected_sha256"] != pin or (
        primary["experiment_id"] in PINS and PINS[primary["experiment_id"]] != pin
    ):
        raise ValueError("producer_anchor_drift")
    expected = [(r["path"], r["sha256"]) for r in requirements(primary)]
    observed = [(r["original_path"], r["expected_sha256"]) for r in bundle["rows"][3:]]
    if expected != observed:
        raise ValueError("closure_membership_drift")
    if (
        side["primary_sha256"] != pin
        or side["primary_path"] != bundle["original_path"]
        or side["report"]["passed"] is not True
        or terminal["publication"]["primary_sha256"] != pin
        or terminal["publication"]["primary_path"] != bundle["original_path"]
    ):
        raise ValueError("terminal_binding_drift")
    return dict(primary)


def adapter(bundle: Json, scratch: Path) -> ModuleType:
    """Import a known reader only from its authenticated captured source closure.

    Package paths select recovered source bytes instead of current aliases.
    Only the four named producer readers can be selected; copied code remains
    inside the bounded worker and is never installed into the repository.
    """
    value = verify(bundle)
    names = {
        8342: "arc_supervisor_frontier_8342",
        8344: "gatemate_ledger_execution_8344",
        8355: "arc_supervisor_frontier_8355",
        8357: "gatemate_ledger_execution_8357",
    }
    selected = "carnot.reporting." + names[value["experiment_id"]]
    for row in bundle["rows"]:
        source = Path(row["original_path"])
        if (
            source.suffix == ".py"
            and source.is_relative_to(ROOT)
            and source.relative_to(ROOT).parts[0] in {"python", "scripts"}
        ):
            target = scratch / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(checked(row["reference"]).read_bytes())
            target.chmod(0o400)
            for name, module in list(sys.modules.items()):
                if getattr(module, "__file__", None) == str(source):
                    sys.modules.pop(name, None)
    for name, module in list(sys.modules.items()):
        if hasattr(module, "__path__") and (
            name == "carnot"
            or name.startswith("carnot.")
            or name == "scripts"
            or name.startswith("scripts.")
        ):
            relative = Path(*name.split("."))
            target = scratch / (
                Path("python") / relative if name.startswith("carnot") else relative
            )
            module.__path__.insert(0, str(target))
    sys.modules.pop(selected, None)
    result = importlib.import_module(selected)

    def relocate(item: Any) -> Any:
        if isinstance(item, Path) and item.is_absolute() and item.is_relative_to(scratch):
            return ROOT / item.relative_to(scratch)
        if isinstance(item, SimpleNamespace):
            for key, child in vars(item).items():
                setattr(item, key, relocate(child))
        elif isinstance(item, list):
            return [relocate(child) for child in item]
        elif isinstance(item, tuple):
            return tuple(relocate(child) for child in item)
        return item

    for module in list(sys.modules.values()):
        if getattr(module, "__file__", "") and str(module.__file__).startswith(str(scratch)):
            for key, item in list(vars(module).items()):
                if not key.startswith("__"):
                    setattr(module, key, relocate(item))
    return result


@contextmanager
def frozen_reads(bundle: Json, scratch: Path, accesses: Json) -> Iterator[None]:
    """Redirect producer paths to sealed bytes and reject mutable authority fallback.

    Existing readers sometimes write reconstructed reductions beside old inputs.
    Redirect those writes to this child's scratch so history remains untouched.
    """
    mapping = {r["original_path"]: r["reference"]["path"] for r in bundle["rows"] if r["available"]}
    original = Path.open
    original_replace, original_mkdir = Path.replace, Path.mkdir
    primary = verify(bundle)
    primitive = next(
        (
            primary[k]
            for k in ["work_reference", "replay_input_reference", "primitive_reference"]
            if primary.get(k)
        )
    )
    work_root = Path(primitive["path"]).parent

    def relocated(path: Path, *args: Any, **kwargs: Any) -> Any:
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        label = str(path.absolute())
        if not any(flag in mode for flag in ["w", "a", "+"]):
            if label in mapping:
                return original(Path(mapping[label]), *args, **kwargs)
            if path.name in {
                "research-roadmap-vNEXT.md",
                "research-roadmap.yaml",
                "research-roadmap-next.yaml",
            }:
                if not path.absolute().is_relative_to(ROOT) and path.is_file():
                    with original(path, "rb") as stream:
                        digest = "sha256:" + hashlib.sha256(stream.read()).hexdigest()
                    if digest in {r["expected_sha256"] for r in bundle["rows"] if r["available"]}:
                        return original(path, *args, **kwargs)
                accesses["count"] += 1
                raise ValueError("mutable_authority_access:" + label)
        elif path.absolute().is_relative_to(work_root):
            target = scratch / path.absolute().relative_to(work_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            return original(target, *args, **kwargs)
        return original(path, *args, **kwargs)

    def moved(path: Path, target: Any) -> Path:
        if path.absolute().is_relative_to(work_root):
            return original_replace(
                scratch / path.absolute().relative_to(work_root),
                scratch / Path(target).absolute().relative_to(work_root),
            )
        return original_replace(path, target)

    def directory(path: Path, *args: Any, **kwargs: Any) -> None:
        target = (
            scratch / path.absolute().relative_to(work_root)
            if path.absolute().is_relative_to(work_root)
            else path
        )
        original_mkdir(target, *args, **kwargs)

    with (
        patch.object(Path, "open", relocated),
        patch.object(Path, "replace", moved),
        patch.object(Path, "mkdir", directory),
    ):
        yield
