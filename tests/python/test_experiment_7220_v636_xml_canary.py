"""Tests for the bounded Qwen3 XML parser canary receipt.

Spec refs: REQ-ARC-WMTE-7220 and SCENARIO-ARC-WMTE-7220-*.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot import experiment_7220_v636_xml_canary as exp


def _missing_import(name: str) -> object:
    raise ModuleNotFoundError(name)


def _version(name: str) -> str:
    return {"vllm": "0.test", "vllm-gguf-plugin": "1.test"}[name]


def _quant(tmp_path: Path) -> Path:
    snapshot = tmp_path / "snapshots" / "revision-7220"
    snapshot.mkdir(parents=True)
    blob = tmp_path / "blobs" / ("a" * 64)
    blob.parent.mkdir()
    blob.write_bytes(b"exact qwen quant fixture")
    link = snapshot / "Qwen3.8-27B-Q4_K_M.gguf"
    link.symlink_to(blob)
    return link


def _gpu_receipt() -> dict:
    return exp.parse_gpu_receipt(
        b"1, GPU-test, NVIDIA GeForce RTX 3090, 24576 MiB, 4 MiB, 0 %\n",
        b"",
    )


# REQ-ARC-WMTE-7220 / SCENARIO-ARC-WMTE-7220-PACKAGE-BLOCK.
def test_package_preflight_names_vllm_as_the_first_terminal_block() -> None:
    rows = exp.package_preflight(_missing_import, _version)

    assert [row["package"] for row in rows] == ["vllm", "vllm-gguf-plugin"]
    assert all(row["importable"] is False for row in rows)
    assert exp.first_package_block(rows) == {
        "failed_check": "vllm_import",
        "upstream": ".venv",
        "field": "vllm_importable",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
    }


# REQ-ARC-WMTE-7220: installed versions remain evidence, not import substitutes.
def test_package_preflight_records_successful_imports_and_versions() -> None:
    rows = exp.package_preflight(lambda name: object(), _version)

    assert [row["version"] for row in rows] == ["0.test", "1.test"]
    assert all(row["importable"] is True for row in rows)
    assert exp.first_package_block(rows) is None


# REQ-ARC-WMTE-7220: an import and its distribution metadata are separate evidence.
def test_package_preflight_keeps_an_import_when_distribution_metadata_is_missing() -> None:
    def missing_metadata(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    rows = exp.package_preflight(lambda name: object(), missing_metadata)

    assert all(row["importable"] is True for row in rows)
    assert all(row["version"] == "distribution_metadata_missing" for row in rows)


# REQ-ARC-WMTE-7220: the selected device must be idle before lease acquisition.
def test_gpu_receipt_uses_exact_device_and_compute_process_bytes() -> None:
    receipt = _gpu_receipt()

    assert receipt["selected_gpu_index"] == 1
    assert receipt["selected_gpu_uuid"] == "GPU-test"
    assert receipt["task_ownable"] is True
    assert receipt["compute_processes"] == []
    assert receipt["query_sha256"] == exp.sha256_bytes(
        b"1, GPU-test, NVIDIA GeForce RTX 3090, 24576 MiB, 4 MiB, 0 %\n"
    )


# REQ-ARC-WMTE-7220: a process on GPU 1 makes the device unavailable.
def test_gpu_receipt_rejects_busy_or_malformed_observations() -> None:
    busy = exp.parse_gpu_receipt(
        b"1, GPU-test, NVIDIA GeForce RTX 3090, 24576 MiB, 1200 MiB, 50 %\n",
        b"GPU-test, 44, python, 1196 MiB\n",
    )
    malformed = exp.parse_gpu_receipt(b"not,csv\n", b"")

    assert busy["task_ownable"] is False
    assert busy["compute_processes"][0]["pid"] == 44
    assert malformed["task_ownable"] is False
    assert malformed["parse_error"] == "gpu_1_row_missing_or_malformed"


# REQ-ARC-WMTE-7220: the native GPU probe is bounded and captures exact stdout.
def test_default_gpu_reader_runs_both_bounded_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs == {"check": True, "capture_output": True, "timeout": 30}
        return SimpleNamespace(stdout=b"query" if len(calls) == 1 else b"apps")

    monkeypatch.setattr(exp.subprocess, "run", fake_run)

    assert exp.read_gpu_bytes() == (b"query", b"apps")
    assert len(calls) == 2


# REQ-ARC-WMTE-7220: the snapshot revision and content hash bind the quant.
def test_quant_identity_follows_the_cache_link_and_hashes_exact_bytes(tmp_path: Path) -> None:
    link = _quant(tmp_path)

    receipt = exp.quant_identity(link)

    assert receipt["cached_path"] == str(link)
    assert receipt["resolved_path"] == str(link.resolve())
    assert receipt["revision"] == "revision-7220"
    assert receipt["sha256"] == exp.sha256_file(link)
    assert receipt["quantization"] == "Q4_K_M"


# REQ-ARC-WMTE-7220: registry lookup never invents a model path.
def test_default_quant_resolver_returns_only_the_registry_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exp, "cached_current_model", lambda **kwargs: None)
    assert exp.resolve_quant_path() is None

    monkeypatch.setattr(
        exp,
        "cached_current_model",
        lambda **kwargs: {"model_path": "/cache/exact.gguf"},
    )
    assert exp.resolve_quant_path() == "/cache/exact.gguf"


# REQ-ARC-WMTE-7220 / SCENARIO-ARC-WMTE-7220-RECEIPT.
def test_blocked_artifact_keeps_four_censored_units_and_validates(tmp_path: Path) -> None:
    quant = exp.quant_identity(_quant(tmp_path))
    packages = exp.package_preflight(_missing_import, _version)
    artifact = exp.base_artifact(exp.RUN_DATE, "test-host", "2026-09-11T12:00:00Z")

    result = exp.finish_package_block(
        artifact,
        packages=packages,
        gpu=_gpu_receipt(),
        quant=quant,
        source_hashes={"source": "sha256:source"},
        raw_evidence=[{"path": "raw", "sha256": "sha256:raw"}],
        completed_at="2026-09-11T12:00:01Z",
        duration_s=0.5,
    )

    assert result["status"] == "blocked"
    assert result["honest_verdict"] == "blocked_vllm_not_installed"
    assert result["verdict_class"] == "blocked"
    assert result["inference_substrate"] == "blocked_no_run"
    assert result["inference_substrate_class"] == "blocked_no_run"
    assert result["MODEL_SPECS"] == [exp.MANDATED_MODEL_SPEC]
    assert result["model_count"] == 1
    assert result["model_invoked"] is False
    assert result["xml_canary_complete_score"] == 0
    assert result["xml_transport_ready_score"] == 0
    assert result["parser_rows"] == []
    assert result["sample_size_budget"] == {
        "planned": 4,
        "attempted": 0,
        "completed": 0,
        "censored": 4,
        "independent_units": 4,
        "exclusions": [],
    }
    assert len(result["rows"]) == 4
    assert all(row["abstention"] is True for row in result["rows"])
    assert exp.validate_artifact(result) == []


# REQ-ARC-WMTE-7220: the blocked finalizer cannot relabel a launchable host.
def test_blocked_finalizer_requires_an_actual_package_failure(tmp_path: Path) -> None:
    with pytest.raises(exp.LiveExecutionRequired):
        exp.finish_package_block(
            exp.base_artifact(exp.RUN_DATE, "host", "time"),
            packages=exp.package_preflight(lambda name: object(), _version),
            gpu=_gpu_receipt(),
            quant=exp.quant_identity(_quant(tmp_path)),
            source_hashes={},
            raw_evidence=[],
            completed_at="time",
            duration_s=0.1,
        )


# REQ-ARC-WMTE-7220 / SCENARIO-ARC-WMTE-7220-RECEIPT.
@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda item: item.__setitem__("model_invoked", True), "blocked_model_invoked"),
        (
            lambda item: item["sample_size_budget"].__setitem__("censored", 3),
            "blocked_denominators",
        ),
        (lambda item: item.__setitem__("parser_rows", [{}]), "blocked_parser_rows"),
        (lambda item: item.__setitem__("MODEL_SPECS", []), "model_spec_mismatch"),
        (
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
            "reproducibility_checksum",
        ),
        (
            lambda item: item.__setitem__("field_principles", {}),
            "field_principles",
        ),
        (
            lambda item: item.__setitem__("xml_canary_complete_score", 1),
            "blocked_scores",
        ),
        (
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
            "blocked_substrate",
        ),
        (
            lambda item: item.__setitem__("gate_check_summary", {}),
            "gate_check_summary",
        ),
    ],
)
def test_validator_rejects_blocked_receipt_mutations(
    tmp_path: Path, mutation: object, expected: str
) -> None:
    result = exp.finish_package_block(
        exp.base_artifact(exp.RUN_DATE, "test-host", "2026-09-11T12:00:00Z"),
        packages=exp.package_preflight(_missing_import, _version),
        gpu=_gpu_receipt(),
        quant=exp.quant_identity(_quant(tmp_path)),
        source_hashes={"source": "sha256:source"},
        raw_evidence=[],
        completed_at="2026-09-11T12:00:01Z",
        duration_s=0.5,
    )
    mutation(result)  # type: ignore[operator]
    if expected != "reproducibility_checksum":
        result["reproducibility_checksum"] = exp.artifact_checksum(result)

    assert expected in exp.validate_artifact(result)


# REQ-ARC-WMTE-7220: the current missing package path writes raw and terminal data.
def test_run_experiment_stops_before_tokenizer_lease_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quant_path = _quant(tmp_path)
    output = tmp_path / "results" / "artifact.json"
    raw_dir = tmp_path / "results" / "raw"
    checkpoint = tmp_path / "results" / "checkpoints" / "running.json"
    times = iter([10.0, 10.25])
    utc_times = iter(["2026-09-11T12:00:00Z", "2026-09-11T12:00:01Z"])
    forbidden_calls: list[str] = []
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)

    def _forbidden_live_canary(**_kwargs: object) -> dict:
        forbidden_calls.append("live_canary")
        raise AssertionError("live canary must not run when a package failed to import")

    result = exp.run_experiment(
        root=exp.REPO_ROOT,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_date=exp.RUN_DATE,
        importer=_missing_import,
        version_reader=_version,
        gpu_reader=lambda: (
            b"1, GPU-test, NVIDIA GeForce RTX 3090, 24576 MiB, 4 MiB, 0 %\n",
            b"",
        ),
        quant_resolver=lambda: str(quant_path),
        live_canary_runner=_forbidden_live_canary,
        clock=lambda: next(times),
        utc_reader=lambda: next(utc_times),
    )

    assert forbidden_calls == []
    assert result["honest_verdict"] == "blocked_vllm_not_installed"
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "running"
    assert sorted(path.name for path in raw_dir.iterdir()) == [
        "gpu_compute_apps.csv",
        "gpu_query.csv",
        "package_imports.json",
        "quant_identity.json",
    ]
    assert exp.validate_artifact(output) == []


# REQ-ARC-WMTE-7220: a launchable preflight takes the real live path, not a forged block.
def test_run_experiment_takes_the_live_path_when_packages_are_present(
    tmp_path: Path,
) -> None:
    output = tmp_path / "final.json"
    live_calls: list[dict] = []

    def _fake_live_canary(**kwargs: object) -> dict:
        live_calls.append(kwargs)
        return {
            "lease_acquired": True,
            "lease_error": None,
            "server_started": True,
            "server_pid": 12345,
            "health": {"healthy": True, "reason": "healthy", "elapsed_s": 12.0},
            "parser_rows": [
                {
                    "tool_name": name,
                    "ok": True,
                    "error": None,
                    "tool_calls": [{"function": {"name": name}}],
                    "populated": True,
                    "finish_reason": "tool_calls",
                    "content_preview": "",
                    "elapsed_s": 1.0,
                    "response_sha256": "sha256:" + name,
                }
                for name in exp.EXPECTED_TOOL_NAMES
            ],
            "vram_resident_mb": 16000,
            "vram_after_mb": 200,
            "exit_code": 0,
            "unload_observed": True,
            "phase_reached": "terminal_complete",
        }

    result = exp.run_experiment(
        root=exp.REPO_ROOT,
        output_path=output,
        raw_dir=tmp_path / "raw",
        checkpoint_path=tmp_path / "checkpoint.json",
        run_date=exp.RUN_DATE,
        importer=lambda name: object(),
        version_reader=_version,
        gpu_reader=lambda: (
            b"1, GPU-test, NVIDIA GeForce RTX 3090, 24576 MiB, 4 MiB, 0 %\n",
            b"",
        ),
        quant_resolver=lambda: str(_quant(tmp_path / "quant")),
        live_canary_runner=_fake_live_canary,
    )

    assert len(live_calls) == 1
    assert result["honest_verdict"] == "complete_positive_xml_transport_confirmed"
    assert result["xml_transport_ready_score"] == 1
    assert result["xml_canary_complete_score"] == 1
    assert len(result["parser_rows"]) == len(exp.EXPECTED_TOOL_NAMES)
    assert output.exists()
    assert exp.validate_artifact(result) == []


# REQ-ARC-WMTE-7220: a partial or absent transport result is reported honestly, not upgraded.
@pytest.mark.parametrize(
    ("live_overrides", "expected_verdict", "expected_score"),
    [
        ({"lease_acquired": False, "phase_reached": None}, "blocked_gpu_lease_unavailable", 0),
        ({"phase_reached": "terminal_blocked"}, "blocked_vllm_server_startup_failed", 0),
    ],
)
def test_finish_live_block_reports_lease_and_server_failures_honestly(
    tmp_path: Path, live_overrides: dict, expected_verdict: str, expected_score: int
) -> None:
    base_live = {
        "lease_acquired": True,
        "lease_error": None,
        "server_started": True,
        "server_pid": 1,
        "health": {"healthy": False, "reason": "startup_timeout", "elapsed_s": 400.0},
        "parser_rows": [],
        "vram_resident_mb": None,
        "vram_after_mb": 100,
        "exit_code": 1,
        "unload_observed": True,
        "phase_reached": "terminal_blocked",
    }
    live = {**base_live, **live_overrides}
    result = exp.finish_live_block(
        exp.base_artifact(exp.RUN_DATE, "test-host", "2026-09-11T12:00:00Z"),
        packages=exp.package_preflight(lambda name: object(), _version),
        gpu=_gpu_receipt(),
        quant=exp.quant_identity(_quant(tmp_path)),
        live=live,
        source_hashes={"source": "sha256:source"},
        raw_evidence=[],
        completed_at="2026-09-11T12:00:01Z",
        duration_s=1.0,
    )
    assert result["honest_verdict"] == expected_verdict
    assert result["xml_transport_ready_score"] == expected_score
    assert result["model_invoked"] is bool(live["server_started"])


# REQ-ARC-WMTE-7220: invalid dates and missing immutable inputs fail before a receipt.
def test_run_experiment_rejects_wrong_date_and_missing_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_date"):
        exp.run_experiment(run_date="20260910")
    with pytest.raises(FileNotFoundError, match="required source paths"):
        exp.run_experiment(
            root=tmp_path,
            output_path=tmp_path / "final.json",
            raw_dir=tmp_path / "raw",
            checkpoint_path=tmp_path / "checkpoint.json",
        )


# REQ-ARC-WMTE-7220: an exact cache miss fails before package or tokenizer work.
def test_run_experiment_rejects_a_quant_cache_miss(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cached unsloth/Qwen3.8-27B-GGUF"):
        exp.run_experiment(
            root=exp.REPO_ROOT,
            output_path=tmp_path / "final.json",
            raw_dir=tmp_path / "raw",
            checkpoint_path=tmp_path / "checkpoint.json",
            gpu_reader=lambda: (
                b"1, GPU-test, NVIDIA GeForce RTX 3090, 24576 MiB, 4 MiB, 0 %\n",
                b"",
            ),
            quant_resolver=lambda: None,
        )


# REQ-ARC-WMTE-7220: validation failure prevents the final atomic write.
def test_run_experiment_refuses_an_invalid_terminal_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "validate_artifact", lambda value: ["forced_error"])
    with pytest.raises(ValueError, match="terminal artifact invalid"):
        exp.run_experiment(
            root=exp.REPO_ROOT,
            output_path=tmp_path / "final.json",
            raw_dir=tmp_path / "raw",
            checkpoint_path=tmp_path / "checkpoint.json",
            importer=_missing_import,
            version_reader=_version,
            gpu_reader=lambda: (
                b"1, GPU-test, NVIDIA GeForce RTX 3090, 24576 MiB, 4 MiB, 0 %\n",
                b"",
            ),
            quant_resolver=lambda: str(_quant(tmp_path / "quant")),
        )

    assert not (tmp_path / "final.json").exists()


# REQ-ARC-WMTE-7220: the CLI forwards the frozen date and prints the failed gate.
def test_main_reports_the_terminal_gate(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    observed: list[str] = []

    def fake_run(*, run_date: str) -> dict:
        observed.append(run_date)
        return {"gate_check_summary": {"failed_check": "vllm_import"}}

    monkeypatch.setattr(exp, "run_experiment", fake_run)

    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert observed == [exp.RUN_DATE]
    assert "vllm_import" in capsys.readouterr().out  # type: ignore[attr-defined]


# REQ-ARC-WMTE-7220: validation reports structural input errors directly.
def test_validator_rejects_non_mapping_missing_field_and_wrong_date() -> None:
    assert exp.validate_artifact([]) == ["artifact_mapping_required"]
    assert exp.validate_artifact({}) == ["missing_required_field:MODEL_SPECS"]
    artifact = exp.base_artifact("wrong", "host", "time")
    artifact["status"] = "blocked"
    artifact["honest_verdict"] = "blocked_vllm_not_installed"
    artifact["verdict_class"] = "blocked"
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)

    assert "run_date" in exp.validate_artifact(artifact)
