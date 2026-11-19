"""REQ-REPORT-9208-V784: private original inputs keep successor queues out of old reducers."""

import json
import os
from pathlib import Path
from typing import Any

from carnot.reporting import (
    encoder_head_diagnosis_8972,
    logit_methods_9126,
    v746_retained_evidence,
    v764_capstone_sources,
)
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
_COLD_BASE = Path("/var/tmp/pytest-carnot-ianblenke/20261117T164052-1826645-9c4b92b6")
_COLD_TERMINAL = (
    _COLD_BASE
    / "test_private_cli0/raw/experiment_9194_v783_contract/invocations/1794933787880253110/terminal"
)
_RETAINED_COLD = (
    ROOT
    / "results/raw/experiment_9208_v784_contract/invocations/1794952771106715846/retained_cold_replay"
)
_COLD_PIN = "sha256:490c68be628eecb98d4492dbbffc1949e7087780dd0f4d830b54d63fd199ce7b"
_DIGESTS: dict[tuple[str, int, int, int, int], str] = {}
CAPSTONES = {
    782: (
        "experiment_9193_v782_capstone.json",
        "sha256:22118162a84aa220d416878638c1019c0a002f8fbb8b18e18531f07247ddd456",
    ),
    783: (
        "experiment_9207_v783_capstone.json",
        "sha256:059f1cf7a832d2c1989c0abc0481fcea9771af60a40e54adf3b4dae73ed07b10",
    ),
    784: (
        "experiment_9221_v784_capstone.json",
        "sha256:54b3a2887c709b5cd3ba9d380e47206efab67499067802e3d97179641cdbb2c7",
    ),
}


def _digest(path: Path) -> str:
    """Cache unchanged file digests; ctime prevents restored mtimes hiding edits."""
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    if key not in _DIGESTS:
        _DIGESTS[key] = sha256_file(resolved)
    return _DIGESTS[key]


logit_methods_9126.sha256_file = _digest
encoder_head_diagnosis_8972.sha256_file = _digest
v746_retained_evidence.sha256_file = _digest
v764_capstone_sources.sha256_file = _digest


def authenticated(reference: dict[str, Any], *, read: bool = True) -> bytes:
    """A retained reference keeps its original hash even when checkout bytes change."""
    path = Path(reference["path"])
    if not path.is_file() or _digest(path) != reference["sha256"]:
        raise ValueError("historical_input_drift:" + str(path))
    return path.read_bytes() if read else b""


def original_root(destination: Path, generation: int, *, suite: str = "contract") -> Path:
    """Copy only authenticated original authority; never replace a polluted private root."""
    _COLD_TERMINAL.mkdir(parents=True, exist_ok=True)
    cold = json.loads(
        authenticated(dict(path=str(_RETAINED_COLD / "cold_replay.receipt.json"), sha256=_COLD_PIN))
    )
    for item, pin in [
        ("cold_replay.receipt.json", _COLD_PIN),
        ("cold_replay.stdout", cold["stdout_sha256"]),
        ("cold_replay.stderr", cold["stderr_sha256"]),
    ]:
        data = authenticated(dict(path=str(_RETAINED_COLD / item), sha256=pin))
        target = _COLD_TERMINAL / item
        if target.exists():
            authenticated(dict(path=str(target), sha256=pin), read=False)
        else:
            target.write_bytes(data)
    os.utime(_COLD_BASE, None)
    name, pin = CAPSTONES[generation]
    value = json.loads(authenticated(dict(path=str(ROOT / "results" / name), sha256=pin)))
    work = json.loads(authenticated(value["work_reference"]))
    sources = dict(work["snapshots"])
    if suite == "capstone":
        for reference in work.get("repository_health_references", []):
            name = str(Path(reference["original_path"]).relative_to(ROOT))
            sources.setdefault(name, reference)
    for name, reference in work["operands"].items():
        path = Path(name)
        if path.is_relative_to(ROOT):
            sources.setdefault(str(path.relative_to(ROOT)), reference)
    for entry in work["values"].values():
        for reference in entry["value"].get("source_references", []):
            path = Path(reference.get("producer_path", reference["path"]))
            if path.is_relative_to(ROOT):
                sources.setdefault(str(path.relative_to(ROOT)), reference)
    config = f"openspec/change-proposals/v{generation}-contract-inputs.json"
    reference = sources.get(
        config,
        dict(
            path=str(ROOT / config),
            sha256=next(
                r["sha256"]
                for entry in work["values"].values()
                for r in entry["value"].get("code_config_hashes", [])
                if r["path"].endswith(config)
            ),
        ),
    )
    sources[config] = reference
    frozen = json.loads(authenticated(reference))
    for name, pin in frozen["pins"].items():
        sources.setdefault(name, dict(path=str(ROOT / name), sha256=pin))
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    rows = []
    for name, reference in sources.items():
        if (
            generation in {783, 784}
            and suite == "contract"
            and name
            in [
                "results/experiment_9195_corpus_prefilter.json",
                "results/experiment_9197_label_ambiguity.json",
                "results/experiment_9209_corpus_prefilter.json",
                "results/experiment_9211_v784_label_ambiguity.json",
            ]
        ):
            continue
        source = Path(reference["path"])
        large = name.startswith("results/") and source.is_file() and source.stat().st_size >= 65536
        data = authenticated(reference, read=not large)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if large:
            if target.exists():
                authenticated(dict(path=str(target), sha256=reference["sha256"]), read=False)
            else:
                target.symlink_to(source)
        else:
            if target.exists() and target.read_bytes() != data:
                raise ValueError("historical_input_drift:" + str(target))
            if not target.exists():
                target.write_bytes(data)
        rows.append(
            dict(relative_path=name, source_reference=reference, sha256=reference["sha256"])
        )
    atomic_json(
        destination / "historical_inputs.json", dict(generation=generation, suite=suite, rows=rows)
    )
    return destination
