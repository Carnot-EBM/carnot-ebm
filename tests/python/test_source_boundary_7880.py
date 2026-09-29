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


def test_preflight_authenticates_originals_and_missing_operand(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    assert scope["coverage_include"].split(",") == [
        str((producer.ROOT / p).resolve()) for p in scope["owned_files"]
    ]
    coverage_commands = [item for item in scope["commands"] if item["name"].startswith("coverage_")]
    assert len(coverage_commands) == 3
    assert all(
        f"--include={scope['coverage_include']}" in item["argv"] for item in coverage_commands
    )
    assert not any(
        "tests/python" == argument for item in scope["commands"] for argument in item["argv"]
    )


def test_custody_keeps_public_and_evaluator_shards_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
            rows.append(
                {
                    "family_id": family,
                    "role": role,
                    "source_group": f"group-{index}",
                    "source_sha256": "source",
                    "answer_sha256": "answer",
                    "complete_source": source,
                    "complete_response": answer,
                    "label_provenance": {"human_label": 1, "annotation_byte_offsets": [[0, 1]]},
                    "exclusion_reasons": [],
                    "arm": "public_projection",
                    "seed": 68480,
                    "status": "completed",
                }
            )
            public.append(
                {
                    "family_id": family,
                    "source_bytes": source.encode().hex(),
                    "answer_bytes": answer.encode().hex(),
                }
            )
            index += 1
    monkeypatch.setattr(
        producer.acquisition, "acquire", lambda _manifest, _progress: (rows, public)
    )
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    artifact = producer.base(0.0, sources, [])
    producer.custody(artifact, manifest, license_data, tmp_path, 0.0)
    assert artifact["sample_size_budget"]["independent"] == 640
    assert artifact["policy_subroles"] == {"policy_design": 32, "calibration_replay": 32}
    cohort = json.loads(Path(artifact["cohort_manifest_path"]).read_text())
    assert cohort["rows"][0]["complete_source"] == "Source 0."
    assert "human_label" not in (tmp_path / "raw/public.jsonl").read_text()
    assert "human_label" in (tmp_path / "raw/evaluator.jsonl").read_text()
    cohort["rows"] = []
    Path(artifact["cohort_manifest_path"]).write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match="immutable_cohort_collision"):
        producer.custody(artifact, manifest, license_data, tmp_path, 0.0)


def test_validation_counts_real_unit_and_cli_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7880-COVERAGE: each isolated file has measured lines."""
    scope = producer.freeze(tmp_path)
    artifact = producer.base(0.0, [], [])
    owned = [producer.ROOT / path for path in scope["owned_files"]]

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        name = spec["name"]
        data_name = {
            "coverage_unit": "unit",
            "coverage_cli_success": "cli_success",
            "coverage_cli_failure": "cli_failure",
            "coverage_combine": "combined",
        }.get(name)
        if data_name:
            data = CoverageData(basename=str(private / f"{data_name}.coverage"))
            files = (
                owned[:1]
                if data_name == "unit"
                else owned[1:]
                if data_name != "combined"
                else owned
            )
            data.add_lines({str(path): {1} for path in files})
            data.write()
        return {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
            "log_sha256": "sha256:fake",
            "resolved_imports": producer.acquisition.qualified_imports(),
        }

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path, 0.0)
    assert not artifact["gate_check_summary"]
    assert len(artifact["validation_receipts"]) == len(scope["commands"]) + 2
    assert all(count > 0 for count in artifact["coverage_statement_counts"]["combined"].values())


def test_validation_rejects_empty_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7880-COVERAGE: an exit-zero empty run is a failure."""
    scope = producer.freeze(tmp_path)
    artifact = producer.base(0.0, [], [])

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        name = spec["name"]
        if name.startswith("coverage_"):
            data_name = {
                "coverage_unit": "unit",
                "coverage_cli_success": "cli_success",
                "coverage_cli_failure": "cli_failure",
            }[name]
            CoverageData(basename=str(private / f"{data_name}.coverage")).write()
        return {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
            "log_sha256": "sha256:fake",
            "resolved_imports": producer.acquisition.qualified_imports(),
        }

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path, 0.0)
    assert {gate["artifact_field"] for gate in artifact["gate_check_summary"]} == {
        "coverage_statement_counts"
    }
    assert not any(
        receipt["name"] == "coverage_combine" for receipt in artifact["validation_receipts"]
    )


def test_terminal_publishes_exact_checked_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7880-TERMINAL: sidecar hashes the published candidate."""
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    monkeypatch.setattr(producer, "OUTPUT", tmp_path / "result.json")
    artifact = producer.base(0.0, [], [])
    artifact["rows"] = [{"family_id": "one", "role": "fit", "status": "completed"}]

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        return {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
            "log_sha256": "sha256:fake",
            "output_tail": json.dumps({"flagged_count": 0}),
        }

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


@pytest.mark.parametrize(
    "fault",
    [
        "upstream",
        "history",
        "license",
        "source_hash",
        "manifest_missing",
        "manifest_schema",
        "manifest_hash",
    ],
)
def test_preflight_private_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: changed upstream bytes fail custody."""
    for name, original in (
        ("upstream", producer.UPSTREAM),
        ("history", producer.HISTORY),
        ("license", producer.LICENSE),
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(original.read_bytes())
        monkeypatch.setattr(producer, name.upper() if name != "upstream" else "UPSTREAM", path)
    if fault in {"upstream", "history", "license", "source_hash", "manifest_missing"}:
        path = (
            tmp_path / "upstream.json"
            if fault in {"upstream", "source_hash", "manifest_missing"}
            else tmp_path / f"{fault}.json"
        )
        value = json.loads(path.read_text())
        if fault == "upstream":
            value["evidence_view_ready_score"] = 0
        elif fault == "history":
            value["verdict_class"] = "positive"
        elif fault == "license":
            value["license"] = "unknown"
        elif fault == "source_hash":
            value["source_artifact_hashes"][0]["sha256"] = "sha256:changed"
        else:
            value["source_view_manifest_path"] = str(tmp_path / "missing.json")
        path.write_text(json.dumps(value))
    if fault in {"manifest_schema", "manifest_hash"}:
        upstream_path = tmp_path / "upstream.json"
        upstream = json.loads(upstream_path.read_text())
        manifest = Path(upstream["source_view_manifest_path"])
        copied = tmp_path / "manifest.json"
        value = json.loads(manifest.read_text())
        if fault == "manifest_schema":
            value["schema"] = "wrong"
        else:
            value["rows_sha256"] = "sha256:changed"
        copied.write_text(json.dumps(value))
        upstream["source_view_manifest_path"] = str(copied)
        upstream_path.write_text(json.dumps(upstream))
    manifest, _, failures, _ = producer.preflight(0.0)
    assert manifest is None and failures
    assert all(
        {"upstream_id", "path", "hash", "artifact_field", "op", "expected", "observed"}
        <= item.keys()
        for item in failures
    )


def test_owned_child_expected_log_is_sealed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: the child receipt binds closed bytes."""
    log = tmp_path / "closed.log"
    log.write_text("FileNotFoundError: missing\n")
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *_a, **_k: [
            {"log_path": str(log), "exit_code": 1, "timed_out": False, "passed": False}
        ],
    )
    spec = {
        "name": "failure",
        "argv": ["false"],
        "classification": "required",
        "timeout_s": 5,
        "expected_exit": "nonzero",
    }
    receipt = producer.child(spec, 1, tmp_path, 0.0)
    assert receipt["passed"] and receipt["log_sha256"] == producer.sha256_file(
        Path(receipt["log_path"])
    )
    Path(receipt["log_path"]).write_text("changed sealed bytes")
    with pytest.raises(ValueError, match="sealed_log_collision"):
        producer.child(spec, 1, tmp_path, 0.0)


def test_dated_orchestration_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: preflight, custody and dispatch are reached."""
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    monkeypatch.setattr(producer, "HISTORY", tmp_path / "history.json")
    producer.HISTORY.write_text("{}")
    monkeypatch.setattr(producer, "terminal", lambda artifact, *_: artifact)
    with pytest.raises(ValueError, match="run_date_mismatch"):
        producer.run_experiment("wrong")
    monkeypatch.setattr(
        producer,
        "preflight",
        lambda _start: (
            None,
            [],
            [producer.operand("exp7810", tmp_path / "missing", "sha256", "present", None)],
            {},
        ),
    )
    blocked = producer.run_experiment("20260929")
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    monkeypatch.setattr(producer, "preflight", lambda _start: ({"schema": "fixture"}, [], [], {}))
    monkeypatch.setattr(
        producer, "custody", lambda *_: (_ for _ in ()).throw(ValueError("source_family_count:639"))
    )
    blocked = producer.run_experiment("20260929")
    assert blocked["gate_check_summary"][-1]["artifact_field"] == "source_family_count"
    monkeypatch.setattr(producer, "custody", lambda *_: None)
    monkeypatch.setattr(producer, "validate", lambda artifact, *_: artifact.update(validated=True))
    completed = producer.run_experiment("20260929")
    assert completed["validated"]


def test_terminal_rechecks_changed_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-TERMINAL: changed verdict bytes face both checks again."""
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    monkeypatch.setattr(producer, "OUTPUT", tmp_path / "result.json")
    artifact = producer.base(0.0, [], [])
    artifact["rows"] = [{"family_id": "one", "role": "fit", "status": "completed"}]
    calls = []

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        calls.append(spec["name"])
        flagged = len(calls) == 1
        return {
            "name": spec["name"],
            "passed": True,
            "exit_code": 0,
            "log_path": str(tmp_path / "fake.log"),
            "output_tail": json.dumps({"flagged_count": int(flagged)}),
        }

    monkeypatch.setattr(producer, "child", fake_child)
    result = producer.terminal(artifact, 0.0, tmp_path)
    assert result["verdict_class"] == "disqualified"
    assert calls == [
        "adversarial_verify",
        "strict_rows",
        "final_adversarial_verify",
        "final_strict_rows",
    ]
    assert json.loads(producer.OUTPUT.read_text())["source_boundary_ready_score"] == 0


def test_preflight_missing_executable_and_private_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: absent tools and direct fixture paths fail closed."""
    original = producer.ROOT
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    manifest, _, failures, _ = producer.preflight(0.0)
    assert manifest is None
    assert any(item["artifact_field"] == "executable" for item in failures)
    monkeypatch.setattr(producer, "ROOT", original)
    public, feature = tmp_path / "public.jsonl", tmp_path / "feature.jsonl"
    producer.source_projection.write_jsonl(
        public, [{"family_id": "fixture", "source_bytes": b"A.".hex(), "answer_bytes": b"A.".hex()}]
    )
    assert producer.main(["--fixture-public", str(public), "--fixture-output", str(feature)]) == 0
    producer.fixture_cli(public, feature)


def test_custody_rejects_missing_family_and_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-CUSTODY: no roster or import alias is accepted."""
    artifact = producer.base(0.0, [], [])
    monkeypatch.setattr(producer.acquisition, "acquire", lambda *_: ([], []))
    with pytest.raises(ValueError, match="source_family_count"):
        producer.custody(artifact, {}, {}, tmp_path, 0.0)
    rows = []
    for role, count in producer.acquisition.ROLES.items():
        rows.extend({"family_id": f"{role}-{index}", "role": role} for index in range(count))
    monkeypatch.setattr(producer.acquisition, "acquire", lambda *_: (rows, []))
    monkeypatch.setattr(producer.acquisition, "qualified_imports", lambda: {"bad": "alias"})
    with pytest.raises(ValueError, match="resolved_imports_invalid"):
        producer.custody(artifact, {}, {}, tmp_path, 0.0)


def test_validation_rejects_failed_required_and_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: failed child cannot be hidden by coverage."""
    scope = producer.freeze(tmp_path)
    artifact = producer.base(0.0, [], [])

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        if spec["name"].startswith("coverage_"):
            names = {
                "coverage_unit": "unit",
                "coverage_cli_success": "cli_success",
                "coverage_cli_failure": "cli_failure",
            }
            data = CoverageData(basename=str(private / f"{names[spec['name']]}.coverage"))
            data.add_lines({str(producer.ROOT / scope["owned_files"][0]): {1}})
            data.write()
        return {
            "name": spec["name"],
            "passed": spec["name"] != "worktree_imports",
            "exit_code": 1 if spec["name"] == "worktree_imports" else 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
            "resolved_imports": {"bad": "alias"},
        }

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path, 0.0)
    assert {item["artifact_field"] for item in artifact["gate_check_summary"]} >= {
        "worktree_imports",
        "resolved_imports",
    }


def test_terminal_invalid_report_and_final_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-TERMINAL: failed final checks prevent publication."""
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")
    monkeypatch.setattr(producer, "OUTPUT", tmp_path / "result.json")
    artifact = producer.base(0.0, [], [])
    calls = []

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        calls.append(spec["name"])
        return {
            "name": spec["name"],
            "passed": len(calls) not in (2, 4),
            "exit_code": 1 if len(calls) in (2, 4) else 0,
            "log_path": str(tmp_path / "fake.log"),
            "output_tail": "malformed",
        }

    monkeypatch.setattr(producer, "child", fake_child)
    with pytest.raises(ValueError, match="final_candidate_verification_failed"):
        producer.terminal(artifact, 0.0, tmp_path)
    assert not producer.OUTPUT.exists()
    assert artifact["flagged_adversarial"]
    assert artifact["gate_check_summary"][-1]["artifact_field"] == "strict_rows"


def test_coverage_report_failure_is_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7892-VALIDATION: a failed report closes the gate."""
    scope = producer.freeze(tmp_path)
    artifact = producer.base(0.0, [], [])
    owned = [producer.ROOT / name for name in scope["owned_files"]]

    def fake_child(spec: dict, index: int, private: Path, start: float) -> dict:
        name = spec["name"]
        shard = {
            "coverage_unit": "unit",
            "coverage_cli_success": "cli_success",
            "coverage_cli_failure": "cli_failure",
        }.get(name)
        if shard:
            data = CoverageData(basename=str(private / f"{shard}.coverage"))
            data.add_lines({str(path): {1} for path in owned})
            data.write()
        return {
            "name": name,
            "passed": name != "coverage_report",
            "exit_code": 1 if name == "coverage_report" else 0,
            "timed_out": False,
            "log_path": str(tmp_path / "fake.log"),
            "resolved_imports": producer.acquisition.qualified_imports(),
        }

    monkeypatch.setattr(producer, "child", fake_child)
    producer.validate(artifact, scope, tmp_path, 0.0)
    assert artifact["gate_check_summary"][-1]["artifact_field"] == "coverage_report"
