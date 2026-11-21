"""REQ-REPORT-9208-V784: private original inputs keep successor queues out of old reducers."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

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
_V785_ARCHIVE_PIN = "sha256:58951a5448e3a428a1e49544e28e8e57cdc2df5a523474a9ae868e82aa600634"
_V787_ARCHIVE_PIN = "sha256:37ac114926d62858baa2f09da27d622e2555ac0ffddba7accc7f7d27155f1479"
_V787_CONFIG_PIN = "sha256:f39315e3029fd9960717da3028f60e432a0db41b8ad0a6b5e3d60e6ec4c59de9"
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
    785: (
        "experiment_9240_v786_capstone.json",
        "sha256:b19a3b647f573f5b74c1dead4584137e25c0301dc47a08efb9cdd3b4e95c9500",
    ),
    786: (
        "experiment_9240_v786_capstone.json",
        "sha256:b19a3b647f573f5b74c1dead4584137e25c0301dc47a08efb9cdd3b4e95c9500",
    ),
    787: (
        "experiment_9240_v786_capstone.json",
        "sha256:b19a3b647f573f5b74c1dead4584137e25c0301dc47a08efb9cdd3b4e95c9500",
    ),
    788: (
        "experiment_9268_v788_capstone.json",
        "sha256:12059727652a55d93a267b186f01cbbb3ed1a043fc79799b67a455aae9eb6d17",
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
_YAML_CACHE: dict[str, Any] = {}
_FAST_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_TRIM_PAIRS: dict[bytes, tuple[bytes, bytes]] = {
    b'  "scan_rows": [\n': (b"  ],\n", b'  "scan_rows": [],\n'),
    b'    "ordered_paths": [\n': (b"    ],\n", b'    "ordered_paths": [],\n'),
    b'  "historical_input_roots": [\n': (b"  ],\n", b'  "historical_input_roots": [],\n'),
    b'  "validation_obligations": {\n': (b"  },\n", b'  "validation_obligations": {},\n'),
    b'  "validation_scope_manifest": {\n': (b"  },\n", b'  "validation_scope_manifest": {},\n'),
}
_COMPACT_CACHE: dict[str, dict[str, Any]] = {}


def _cached_safe_load(stream: Any) -> Any:
    """Cache multi-megabyte completion archives across repeated replay checks."""
    if isinstance(stream, bytes) and len(stream) >= 1_000_000:
        key = hashlib.sha256(stream).hexdigest()
        if key not in _YAML_CACHE:
            _YAML_CACHE[key] = yaml.load(stream, Loader=_FAST_LOADER)
        return _YAML_CACHE[key]
    return yaml.load(stream, Loader=_FAST_LOADER)


yaml.safe_load = _cached_safe_load
_cached_safe_load((ROOT / "research-complete.yaml").read_bytes())


def load_compact_primary(path: Path) -> dict[str, Any]:
    """Stream primary JSON metadata without allocating multi-hundred-megabyte inline tables."""
    digest = _digest(path)
    if digest not in _COMPACT_CACHE:
        out = bytearray()
        skip_until: bytes | None = None
        with path.open("rb") as stream:
            for line in stream:
                if skip_until is not None:
                    if line == skip_until:
                        skip_until = None
                    continue
                hit = _TRIM_PAIRS.get(line)
                if hit is not None:
                    skip_until, replacement = hit
                    out.extend(replacement)
                else:
                    out.extend(line)
        _COMPACT_CACHE[digest] = json.loads(out)
    return dict(json.loads(json.dumps(_COMPACT_CACHE[digest])))


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
    if generation in {785, 786, 787}:
        sources.update(
            {
                "openspec/change-proposals/research-roadmap-v786-preserved-20261120.md": dict(
                    path=str(
                        ROOT
                        / "openspec/change-proposals/research-roadmap-v786-preserved-20261120.md"
                    ),
                    sha256="sha256:a1c32ac1892b59219fe288446020893822350f866781f89a39088b76d1b7c763",
                ),
                "openspec/change-proposals/research-roadmap-v786-queue-preserved-20261120.yaml": dict(
                    path=str(
                        ROOT
                        / "openspec/change-proposals/research-roadmap-v786-queue-preserved-20261120.yaml"
                    ),
                    sha256="sha256:2681afbc366b7282955350ecfbedf5d4b932d8dae58c32ca0ea329672356c8ec",
                ),
                "results/experiment_9240_v786_capstone.json": dict(
                    path=str(ROOT / "results/experiment_9240_v786_capstone.json"),
                    sha256=pin,
                ),
            }
        )
    if generation == 787:
        sources.update(
            {
                "openspec/change-proposals/v787-contract-inputs.json": dict(
                    path=str(ROOT / "openspec/change-proposals/v787-contract-inputs.json"),
                    sha256=_V787_CONFIG_PIN,
                ),
                "openspec/change-proposals/research-roadmap-v787-preserved-20261120.md": dict(
                    path=str(
                        ROOT
                        / "openspec/change-proposals/research-roadmap-v787-preserved-20261120.md"
                    ),
                    sha256="sha256:27e73f05af455de93477bd14f74a6e9c238724a6a5959cc66f33278aeaaea23a",
                ),
                "openspec/change-proposals/research-roadmap-v787-queue-preserved-20261120.yaml": dict(
                    path=str(
                        ROOT
                        / "openspec/change-proposals/research-roadmap-v787-queue-preserved-20261120.yaml"
                    ),
                    sha256="sha256:37442df99c095e1288d96f8e84e0f425838f4f8042d328d6f15b3a00ec78eec2",
                ),
                "results/experiment_9241_v787_contract.json": dict(
                    path=str(ROOT / "results/experiment_9241_v787_contract.json"),
                    sha256="sha256:32cdfef7df5fb0b829ef9eba8d68a44a0e6b7947afbd402d24129dae27a92f5b",
                ),
                "results/experiment_9242_corpus_prefilter.json": dict(
                    path=str(ROOT / "results/experiment_9242_corpus_prefilter.json"),
                    sha256="sha256:9e5cf741802b6c1202c6bbbb5ff3b008929d2fbd39b079ff64a00a2d50e63063",
                ),
                "results/experiment_9244_v787_conformal_risk_protocol.json": dict(
                    path=str(ROOT / "results/experiment_9244_v787_conformal_risk_protocol.json"),
                    sha256="sha256:36890bc36e8a84c4d91b8d8bff2d60698de061938f591dafbf7845c38fe2e693",
                ),
            }
        )
        authenticated(
            json.loads(authenticated(sources["results/experiment_9241_v787_contract.json"]))[
                "work_reference"
            ]
        )
        authenticated(
            json.loads(
                authenticated(sources["results/experiment_9244_v787_conformal_risk_protocol.json"])
            )["work_reference"]
        )
        v787_archive = (
            (ROOT / "research-complete.yaml").read_bytes().split(b"- id: 2026.11.787\n", 1)[0]
        )
        v787_target = _COLD_TERMINAL / "v787_research_complete.yaml"
        if v787_target.exists():
            authenticated(dict(path=str(v787_target), sha256=_V787_ARCHIVE_PIN), read=False)
        else:
            v787_target.write_bytes(v787_archive)
        sources["research-complete.yaml"] = dict(path=str(v787_target), sha256=_V787_ARCHIVE_PIN)
        sources["openspec/change-proposals/research-roadmap-vNEXT.md"] = sources[
            "openspec/change-proposals/research-roadmap-v787-preserved-20261120.md"
        ]
        sources["research-roadmap.yaml"] = sources[
            "openspec/change-proposals/research-roadmap-v787-queue-preserved-20261120.yaml"
        ]
    if generation == 788:
        sources.update(
            {
                "openspec/change-proposals/research-roadmap-v788-preserved-20261121.md": dict(
                    path=str(
                        ROOT
                        / "openspec/change-proposals/research-roadmap-v788-preserved-20261121.md"
                    ),
                    sha256="sha256:c7b13576b7bf8c46c51410dc9905c26924c02e75c4f509ebea3ae8bf31f897cd",
                ),
                "openspec/change-proposals/research-roadmap-v788-queue-preserved-20261121.yaml": dict(
                    path=str(
                        ROOT
                        / "openspec/change-proposals/research-roadmap-v788-queue-preserved-20261121.yaml"
                    ),
                    sha256="sha256:bbf68a85425cde6bf1328dae29db817d2db38b8b26cf809ce139e7d13a8d675b",
                ),
                "results/experiment_9255_v788_contract.json": dict(
                    path=str(ROOT / "results/experiment_9255_v788_contract.json"),
                    sha256="sha256:7c669d5df1331eb5d22211d22bca58db4959b76af6abcaa814ca5de5a5ea24ef",
                ),
                "results/experiment_9256_v788_corpus_prefilter.json": dict(
                    path=str(ROOT / "results/experiment_9256_v788_corpus_prefilter.json"),
                    sha256="sha256:d1ac40d315f6a740c5cb30d853045f6c92ed51d2cebeef14e09b7b945f17481a",
                ),
                "results/experiment_9257_forward_qualification.json": dict(
                    path=str(ROOT / "results/experiment_9257_forward_qualification.json"),
                    sha256="sha256:10bbcaf0b653f5b8b73f55393af7f8b805520316dc211911a089603c01b86ac6",
                ),
                "results/experiment_9258_v788_victris_energy_bot.json": dict(
                    path=str(ROOT / "results/experiment_9258_v788_victris_energy_bot.json"),
                    sha256="sha256:7c66714ad53ba5c88e38a94df68a74ab808f0b1773c3ce1f062d93b927ec5640",
                ),
                "results/experiment_9268_v788_capstone.json": dict(
                    path=str(ROOT / "results/experiment_9268_v788_capstone.json"),
                    sha256=pin,
                ),
            }
        )
        authenticated(work["values"]["exp9255-contract"]["value"]["work_reference"], read=False)
        authenticated(
            work["values"]["exp9258-victris-energy-bot"]["value"]["work_reference"], read=False
        )
        from carnot.reporting import v788_capstone_evidence as _v788_cap_e

        for big_name in [
            "results/experiment_9255_v788_contract.json",
            "results/experiment_9256_v788_corpus_prefilter.json",
        ]:
            big_ref = sources[big_name]
            _v788_cap_e._PROJECTED_CACHE[big_ref["sha256"]] = _v788_cap_e.project(
                load_compact_primary(Path(big_ref["path"]))
            )
        sources["openspec/change-proposals/research-roadmap-vNEXT.md"] = sources[
            "openspec/change-proposals/research-roadmap-v788-preserved-20261121.md"
        ]
        sources["research-roadmap.yaml"] = sources[
            "openspec/change-proposals/research-roadmap-v788-queue-preserved-20261121.yaml"
        ]
    if generation == 786:
        sources["openspec/change-proposals/research-roadmap-vNEXT.md"] = sources[
            "openspec/change-proposals/research-roadmap-v786-preserved-20261120.md"
        ]
        sources["research-roadmap.yaml"] = sources[
            "openspec/change-proposals/research-roadmap-v786-queue-preserved-20261120.yaml"
        ]
    if generation == 785:
        authenticated(
            json.loads(authenticated(sources["results/experiment_9222_v785_contract.json"]))[
                "work_reference"
            ]
        )
        v785_archive = authenticated(sources["research-complete.yaml"]).split(
            b"- id: 2026.11.785\n", 1
        )[0]
        v785_target = _COLD_TERMINAL / "v785_research_complete.yaml"
        if v785_target.exists():
            authenticated(dict(path=str(v785_target), sha256=_V785_ARCHIVE_PIN), read=False)
        else:
            v785_target.write_bytes(v785_archive)
        sources["research-complete.yaml"] = dict(path=str(v785_target), sha256=_V785_ARCHIVE_PIN)
        sources["openspec/change-proposals/research-roadmap-vNEXT.md"] = sources[
            "openspec/change-proposals/research-roadmap-v784-preserved-20261118.md"
        ]
        sources["research-roadmap.yaml"] = sources[
            "openspec/change-proposals/research-roadmap-v785-queue-preserved-20261119.yaml"
        ]
    if suite == "capstone":
        for reference in work.get("repository_health_references", []):
            orig = Path(reference["original_path"])
            name = str(orig.relative_to(ROOT) if orig.is_relative_to(ROOT) else orig)
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
    config = (
        "openspec/change-proposals/v786-capstone-inputs.json"
        if generation == 786
        else f"openspec/change-proposals/v{generation}-contract-inputs.json"
    )
    if config not in sources:
        sources[config] = dict(
            path=str(ROOT / config),
            sha256=next(
                r["sha256"]
                for r in [
                    *(
                        item
                        for entry in work["values"].values()
                        for item in entry["value"].get("code_config_hashes", [])
                    ),
                    *work.get("code", []),
                ]
                if r.get("original_path", r["path"]).endswith(config)
            ),
        )
    reference = sources[config]
    frozen = json.loads(authenticated(reference))
    for name, pin in frozen["pins"].items():
        sources.setdefault(name, dict(path=str(ROOT / name), sha256=pin))
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    rows = []
    for name, reference in sources.items():
        if suite == "contract" and name in {
            783: {
                "results/experiment_9195_corpus_prefilter.json",
                "results/experiment_9197_label_ambiguity.json",
            },
            784: {
                "results/experiment_9195_corpus_prefilter.json",
                "results/experiment_9197_label_ambiguity.json",
                "results/experiment_9209_corpus_prefilter.json",
                "results/experiment_9211_v784_label_ambiguity.json",
            },
            785: {
                "results/experiment_9223_corpus_prefilter.json",
                "results/experiment_9225_v785_selective_deferral_protocol.json",
            },
            787: {
                "results/experiment_9242_corpus_prefilter.json",
                "results/experiment_9244_v787_conformal_risk_protocol.json",
            },
            788: {
                "results/experiment_9256_v788_corpus_prefilter.json",
                "results/experiment_9257_forward_qualification.json",
                "results/experiment_9258_v788_victris_energy_bot.json",
                "results/experiment_9268_v788_capstone.json",
            },
        }.get(generation, set()):
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
