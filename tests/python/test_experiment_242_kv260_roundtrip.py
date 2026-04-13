"""Spec: REQ-SAMPLE-007, SCENARIO-SAMPLE-012, SCENARIO-SAMPLE-013, SCENARIO-SAMPLE-014."""

from __future__ import annotations

import importlib.util
import json
import runpy
import types
from pathlib import Path

import pytest
from carnot.samplers.fpga_ising import SoftwareFPGAOverlay


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_242_kv260_roundtrip.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_242_kv260_roundtrip",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def write_bitfile(path: Path) -> Path:
    path.write_bytes(b"bitstream")
    return path


class ProxyMMIO:
    """SCENARIO-SAMPLE-012: proxy real-MMIO semantics over the software overlay for tests."""

    def __init__(self) -> None:
        self._overlay = SoftwareFPGAOverlay(seed=242)

    def write(self, offset: int, value: int) -> None:
        self._overlay.write(offset, value)

    def read(self, offset: int) -> int:
        return self._overlay.read(offset)


class IdleMMIO:
    """SCENARIO-SAMPLE-013: transport that never raises DONE."""

    def __init__(self) -> None:
        self._memory: dict[int, int] = {}

    def write(self, offset: int, value: int) -> None:
        self._memory[offset] = value

    def read(self, offset: int) -> int:
        return self._memory.get(offset, 0)


def test_run_experiment_records_hardware_roundtrip_latencies(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-012: successful bring-up records hardware-labelled timings."""
    module = load_module()
    bitfile = write_bitfile(tmp_path / "carnot_ising.bit")

    payload = module.run_experiment(
        output_path=tmp_path / "experiment_242_results.json",
        bitfile_path=bitfile,
        overlay_loader=lambda path: ProxyMMIO(),
        auto_overlay_factory=lambda _bitfile: ProxyMMIO(),
    )

    assert payload["experiment"] == 242
    assert payload["run_status"] == "complete"
    assert payload["metadata"]["execution_path"] == "hardware"
    assert payload["metadata"]["hardware_detected"] is True
    assert payload["metadata"]["auto_backend_probe"] == {
        "backend_name": "fpga",
        "using_cpu_fallback": False,
    }
    assert payload["round_trip"]["latency_seconds"]["upload"] >= 0.0
    assert payload["round_trip"]["latency_seconds"]["trigger"] >= 0.0
    assert payload["round_trip"]["latency_seconds"]["readback"] >= 0.0
    assert payload["round_trip"]["sample_shape"] == [4, 128]
    assert payload["blockers"] == []


def test_run_experiment_labels_software_overlay_honestly(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-014: software-overlay bring-up never claims hardware validation."""
    module = load_module()
    bitfile = write_bitfile(tmp_path / "carnot_ising.bit")

    payload = module.run_experiment(
        output_path=tmp_path / "experiment_242_results.json",
        bitfile_path=bitfile,
        overlay_loader=lambda path: SoftwareFPGAOverlay(seed=17),
        auto_overlay_factory=lambda _bitfile: None,
    )

    assert payload["run_status"] == "complete"
    assert payload["metadata"]["execution_path"] == "software_model"
    assert payload["metadata"]["hardware_detected"] is False
    assert payload["metadata"]["auto_backend_probe"] == {
        "backend_name": "cpu_fallback",
        "using_cpu_fallback": True,
    }
    assert payload["round_trip"]["latency_seconds"]["trigger"] >= 0.0
    assert any("software-model" in note for note in payload["metadata"]["notes"])


def test_run_experiment_reports_setup_and_runtime_blockers_honestly(tmp_path: Path) -> None:
    """REQ-SAMPLE-007, SCENARIO-SAMPLE-013: blocked hardware stays blocked in the artifact."""
    module = load_module()

    missing_config = module.run_experiment(
        output_path=tmp_path / "missing_config.json",
        bitfile_path=None,
    )
    assert missing_config["run_status"] == "blocked"
    assert missing_config["metadata"]["execution_path"] == "blocked"
    assert missing_config["blockers"][0]["code"] == "missing_bitfile_config"
    assert "CARNOT_KV260_BITFILE" in missing_config["blockers"][0]["setup_step"]
    assert missing_config["round_trip"] is None
    assert missing_config["metadata"]["auto_backend_probe"] == {
        "backend_name": "cpu_fallback",
        "using_cpu_fallback": True,
    }

    missing_path = module.run_experiment(
        output_path=tmp_path / "missing_path.json",
        bitfile_path=tmp_path / "absent.bit",
    )
    assert missing_path["blockers"][0]["code"] == "bitfile_not_found"

    bitfile = write_bitfile(tmp_path / "carnot_ising.bit")
    missing_mmio = module.run_experiment(
        output_path=tmp_path / "missing_mmio.json",
        bitfile_path=bitfile,
        overlay_loader=lambda path: None,
    )
    assert missing_mmio["run_status"] == "blocked"
    assert missing_mmio["blockers"][0]["code"] == "missing_mmio_endpoint"

    load_error = module.run_experiment(
        output_path=tmp_path / "load_error.json",
        bitfile_path=bitfile,
        overlay_loader=lambda path: (_ for _ in ()).throw(RuntimeError("pynq missing")),
    )
    assert load_error["blockers"][0]["code"] == "overlay_load_failed"
    assert "pynq missing" in load_error["blockers"][0]["error"]

    runtime_blocked = module.run_experiment(
        output_path=tmp_path / "runtime_blocked.json",
        bitfile_path=bitfile,
        overlay_loader=lambda path: IdleMMIO(),
        auto_overlay_factory=lambda _bitfile: None,
    )
    assert runtime_blocked["run_status"] == "blocked"
    assert runtime_blocked["metadata"]["execution_path"] == "blocked"
    assert runtime_blocked["blockers"][0]["code"] == "roundtrip_failed"
    assert "did not complete" in runtime_blocked["blockers"][0]["error"]


def test_main_writes_payload_with_repo_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-007: the Exp 242 CLI writes the checked-in artifact path by default."""
    module = load_module()
    repo = make_repo(tmp_path)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    written_payload = {"experiment": 242, "run_status": "blocked"}

    def fake_run_experiment(**kwargs):
        assert kwargs["output_path"] == repo / "results" / "experiment_242_results.json"
        assert kwargs["bitfile_path"] is None
        return written_payload

    monkeypatch.setattr(module, "run_experiment", fake_run_experiment)

    exit_code = module.main([])

    assert exit_code == 0
    output_path = repo / "results" / "experiment_242_results.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == written_payload


def test_loader_helpers_cover_bound_mmio_repo_root_fallback_and_main_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAMPLE-007: helper edges keep the bring-up script executable and deterministic."""
    module = load_module()
    monkeypatch.delenv("CARNOT_REPO_ROOT", raising=False)
    assert module.get_repo_root() == Path(__file__).resolve().parents[2]

    fake_mmio = ProxyMMIO()

    class FakeOverlay:
        def __init__(self, path: str, download: bool = True) -> None:
            self.path = path
            self.download = download
            self.carnot_ising_0 = types.SimpleNamespace(mmio=fake_mmio)

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(Overlay=FakeOverlay),
    )
    bound = module.default_overlay_loader(tmp_path / "carnot_ising.bit")
    assert bound is not None
    bound.write(12, 34)
    assert bound.read(12) == 34

    class MissingEndpointOverlay:
        def __init__(self, path: str, download: bool = True) -> None:
            self.path = path
            self.download = download
            self.carnot_ising_0 = types.SimpleNamespace()

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(Overlay=MissingEndpointOverlay),
    )
    assert module.default_overlay_loader(tmp_path / "carnot_ising.bit") is None

    repo = make_repo(tmp_path)
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "experiment_242_kv260_roundtrip.py"
    )
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
    monkeypatch.setattr("sys.argv", [str(module_path)])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(module_path), run_name="__main__")

    assert exit_info.value.code == 0
    payload = json.loads((repo / "results" / "experiment_242_results.json").read_text())
    assert payload["run_status"] == "blocked"
