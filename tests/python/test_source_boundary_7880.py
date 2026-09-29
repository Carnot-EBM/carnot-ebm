"""REQ-REPORT-7880: current measured source custody."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from coverage import CoverageData
import pytest

from carnot.reporting import source_boundary_7880 as boundary
from scripts.experiments import experiment_7880_v684_source_boundary as producer


SCRIPT = "scripts/experiments/experiment_7880_v684_source_boundary.py"


def test_policy_split_uses_only_family_ids() -> None:
    """SCENARIO-REPORT-7880-CUSTODY: labels cannot select policy subroles."""
    rows = [{"family_id": f"family-{i}", "role": "policy", "label": i % 2} for i in range(64)]
    original = boundary.policy_subroles(rows)
    changed = boundary.policy_subroles([{**row, "label": 1 - row["label"]} for row in rows])
    assert original == changed
    assert sorted(original.values()).count("policy_design") == 32
    assert sorted(original.values()).count("calibration_replay") == 32
    with pytest.raises(ValueError, match="policy_family_count"):
        boundary.policy_subroles(rows[:-1])


def test_exact_coverage_files_reject_old_empty_pattern(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-COVERAGE: zero measured files cannot qualify."""
    module = tmp_path / "python" / "carnot" / "reporting" / "source_boundary_7880.py"
    cli = tmp_path / "scripts" / "experiments" / "experiment_7880.py"
    for path in (module, cli):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n")
    old = tmp_path / "old.coverage"
    CoverageData(basename=str(old)).write()
    with pytest.raises(ValueError, match="coverage_missing_statements"):
        boundary.measured_counts(old, [module, cli])
    fresh = tmp_path / "fresh.coverage"
    data = CoverageData(basename=str(fresh))
    data.add_lines({str(module): {1}, str(cli): {1}})
    data.write()
    assert boundary.measured_counts(fresh, [module, cli]) == {str(module): 1, str(cli): 1}
    with pytest.raises(ValueError, match="coverage_missing_statements"):
        boundary.measured_counts(fresh, [module, tmp_path / "wrong.py"])


def test_closed_venue() -> None:
    """SCENARIO-REPORT-7880-COVERAGE: the prior host_cpu spelling fails."""
    assert boundary.valid_venue("host")
    assert not boundary.valid_venue("host_cpu")


@pytest.mark.parametrize("fault", ["none", "missing", "malformed", "label", "duplicate"])
def test_real_cli_public_boundary(tmp_path: Path, fault: str) -> None:
    """SCENARIO-REPORT-7880-TERMINAL: script-path routes use public bytes only."""
    directory = tmp_path / "path with spaces"
    directory.mkdir()
    public, output = directory / "public.jsonl", directory / "features.jsonl"
    row = {"family_id": "one", "source_bytes": b"A. B.".hex(), "answer_bytes": b"A.".hex()}
    if fault == "malformed":
        row["source_bytes"] = "not hex"
    if fault == "label":
        row["label"] = 1
    rows = [row, row] if fault == "duplicate" else [row]
    if fault != "missing":
        public.write_text("".join(json.dumps(item) + "\n" for item in rows))
    command = [
        sys.executable,
        "-u",
        SCRIPT,
        "--fixture-public",
        str(public),
        "--fixture-output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    assert (result.returncode == 0) == (fault == "none")
    if fault == "none":
        assert output.is_file()


def test_preflight_authenticates_originals_and_missing_operand(monkeypatch: pytest.MonkeyPatch,
                                                               tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-CUSTODY: originals pass; a missing producer blocks."""
    manifest, sources, failures, license_data = producer.preflight(0.0)
    assert manifest is not None and not failures and len(sources) >= 20
    assert license_data["license"] == "MIT"
    monkeypatch.setattr(producer, "UPSTREAM", tmp_path / "missing.json")
    missing, _, failures, _ = producer.preflight(0.0)
    assert missing is None
    assert failures[0]["upstream_id"] == "exp7810"
    assert failures[0]["artifact_field"] == "sha256"


def test_freeze_declares_absolute_files_before_children(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-COVERAGE: unit and CLI use one exact file set."""
    scope = producer.freeze(tmp_path)
    assert scope["coverage_include"].split(",") == [str((producer.ROOT / p).resolve()) for p in scope["owned_files"]]
    coverage_commands = [item for item in scope["commands"] if item["name"].startswith("coverage_")]
    assert len(coverage_commands) == 3
    assert all(f"--include={scope['coverage_include']}" in item["argv"] for item in coverage_commands)
    assert not any("tests/python" == argument for item in scope["commands"] for argument in item["argv"])


def test_custody_keeps_public_and_evaluator_shards_separate(monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-CUSTODY: original text and annotations stay joined."""
    manifest, sources, failures, license_data = producer.preflight(0.0)
    assert manifest is not None and not failures
    rows = []
    public = []
    index = 0
    for role, count in producer.acquisition.ROLES.items():
        for _ in range(count):
            family = f"family-{index}"
            source = f"Source {index}."
            answer = f"Answer {index}."
            rows.append({"family_id": family, "role": role, "source_group": f"group-{index}",
                "source_sha256": "source", "answer_sha256": "answer", "complete_source": source,
                "complete_response": answer, "label_provenance": {"human_label": 1,
                    "annotation_byte_offsets": [[0, 1]]}, "exclusion_reasons": [],
                "arm": "public_projection", "seed": 68480, "status": "completed"})
            public.append({"family_id": family, "source_bytes": source.encode().hex(),
                           "answer_bytes": answer.encode().hex()})
            index += 1
    monkeypatch.setattr(producer.acquisition, "acquire", lambda _manifest, _progress: (rows, public))
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    artifact = producer.base(0.0, sources, [])
    producer.custody(artifact, manifest, license_data, tmp_path, 0.0)
    assert artifact["sample_size_budget"]["independent"] == 640
    assert artifact["policy_subroles"] == {"policy_design": 32, "calibration_replay": 32}
    cohort = json.loads(Path(artifact["cohort_manifest_path"]).read_text())
    assert cohort["rows"][0]["complete_source"] == "Source 0."
    assert "human_label" not in (tmp_path / "raw/public.jsonl").read_text()
    assert "human_label" in (tmp_path / "raw/evaluator.jsonl").read_text()


def test_validation_counts_real_unit_and_cli_lines(monkeypatch: pytest.MonkeyPatch,
                                                   tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-COVERAGE: each isolated file has measured lines."""
    scope = producer.freeze(tmp_path)
    artifact = producer.base(0.0, [], [])
    owned = [producer.ROOT / path for path in scope["owned_files"]]

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        name = spec["name"]
        data_name = {"coverage_unit": "unit", "coverage_cli_success": "cli_success",
                     "coverage_cli_failure": "cli_failure", "coverage_combine": "combined"}.get(name)
        if data_name:
            data = CoverageData(basename=str(private / f"{data_name}.coverage"))
            files = owned[:1] if data_name == "unit" else owned[1:] if data_name != "combined" else owned
            data.add_lines({str(path): {1} for path in files})
            data.write()
        return {"name": name, "passed": True, "exit_code": 0, "timed_out": False,
            "log_path": str(tmp_path / "fake.log"), "log_sha256": "sha256:fake",
            "resolved_imports": producer.acquisition.qualified_imports()}

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path, 0.0)
    assert not artifact["gate_check_summary"]
    assert len(artifact["validation_receipts"]) == len(scope["commands"]) + 2
    assert all(count > 0 for count in artifact["coverage_statement_counts"]["combined"].values())


def test_validation_rejects_empty_coverage(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-COVERAGE: an exit-zero empty run is a failure."""
    scope = producer.freeze(tmp_path)
    artifact = producer.base(0.0, [], [])

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        name = spec["name"]
        if name.startswith("coverage_"):
            data_name = {"coverage_unit": "unit", "coverage_cli_success": "cli_success",
                         "coverage_cli_failure": "cli_failure"}[name]
            CoverageData(basename=str(private / f"{data_name}.coverage")).write()
        return {"name": name, "passed": True, "exit_code": 0, "timed_out": False,
            "log_path": str(tmp_path / "fake.log"), "log_sha256": "sha256:fake",
            "resolved_imports": producer.acquisition.qualified_imports()}

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path, 0.0)
    assert {gate["artifact_field"] for gate in artifact["gate_check_summary"]} == {"coverage_statement_counts"}
    assert not any(receipt["name"] == "coverage_combine" for receipt in artifact["validation_receipts"])


def test_terminal_publishes_exact_checked_bytes(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-TERMINAL: sidecar hashes the published candidate."""
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    monkeypatch.setattr(producer, "OUTPUT", tmp_path / "result.json")
    artifact = producer.base(0.0, [], [])
    artifact["rows"] = [{"family_id": "one", "role": "fit", "status": "completed"}]

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        return {"name": spec["name"], "passed": True, "exit_code": 0, "timed_out": False,
            "log_path": str(tmp_path / "fake.log"), "log_sha256": "sha256:fake",
            "output_tail": json.dumps({"flagged_count": 0})}

    monkeypatch.setattr(producer, "child", fake_child)
    producer.terminal(artifact, 0.0, tmp_path)
    result = json.loads(producer.OUTPUT.read_text())
    sidecar = json.loads((producer.RAW / "terminal_validation_receipts.json").read_text())
    assert sidecar["candidate_sha256"] == producer.sha256_file(producer.OUTPUT)
    assert result["execution_venue"] == "host"
    assert result["source_boundary_ready_score"] == 1


def test_main_requires_date_or_complete_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7880-TERMINAL: only named public and dated routes run."""
    with pytest.raises(SystemExit):
        producer.main([])
    calls = []
    monkeypatch.setattr(producer, "run_experiment", lambda date: calls.append(date))
    assert producer.main(["--date", "20260929"]) == 0
    assert calls == ["20260929"]
