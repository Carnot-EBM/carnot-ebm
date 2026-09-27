"""REQ-REPORT-7780: V676 capstone source custody and terminal gates."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

import carnot.experiment_7780_v676_capstone as capstone
from carnot.experiment_7767_v676_contract_methods import compare_contract


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def preserved_v676_authorities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run historical V676 checks against the authenticated immutable bytes."""
    roadmap_path = ROOT / "tests/python/fixtures/roadmap_2026_09_676.yaml"
    design_path = ROOT / "openspec/change-proposals/research-roadmap-v676-preserved-20260927.md"
    monkeypatch.setattr(capstone, "DESIGN", design_path)
    monkeypatch.setattr(
        capstone,
        "resolve_authority",
        lambda _root: (roadmap_path, yaml.safe_load(roadmap_path.read_text()), []),
    )


def load_cli():
    """Load the real thin CLI without running its command line entrypoint."""
    spec = importlib.util.spec_from_file_location(
        "exp7780_cli", ROOT / "scripts/experiments/experiment_7780_v676_capstone.py"
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_three_authorities_and_order() -> None:
    """SCENARIO-REPORT-7780-CUSTODY: all fourteen contract rows agree."""
    source = capstone.authority(ROOT)
    assert source["comparison"]["passed"]
    assert len(source["tasks"]) == 14
    assert [task["id"].split("-")[0] for task in source["tasks"]] == [
        f"exp{number}" for number in range(7767, 7781)
    ]
    altered = deepcopy(source["roadmap"])
    altered["tasks"][7]["deliverable"] = "wrong.json"
    assert not compare_contract((ROOT / source["design_path"]).read_text(), altered)["passed"]


def test_current_custody_keeps_missing_and_receipts_separate() -> None:
    """SCENARIO-REPORT-7780-CUSTODY: queue receipts do not fill producers."""
    rows, sources, failures = capstone.account(ROOT, capstone.authority(ROOT)["tasks"])
    assert len(rows) == 14
    assert rows[4]["availability"] == "pre_gate_receipt"
    assert rows[4]["producer_hash"] is None
    assert rows[4]["pre_gate_receipt_hash"] is not None
    assert rows[5]["availability"] == "absent"
    assert rows[-1]["availability"] == "planned_output"
    assert rows[-1]["producer_hash"] is None
    assert len(sources) == 13
    assert any(f["upstream_id"] == "Exp7771" and f["field"] == "producer_exists" for f in failures)
    assert any(f["upstream_id"] == "Exp7775" for f in failures)
    with pytest.raises(ValueError, match="fourteen-task order"):
        capstone.account(ROOT, [])


def test_benefit_gates_and_old_boundaries() -> None:
    """SCENARIO-REPORT-7780-GATES: readiness does not promote benefit."""
    result = capstone.build_artifact(ROOT, {}, [{"name": "focused_pytest", "passed": True}])
    assert result["honest_verdict"] == "complete_blocked_required_v676_evidence"
    assert result["verdict_class"] == "blocked"
    assert result["capstone_complete_score"] == 0
    gates = result["acceptance_gate_results"]
    assert gates["readiness"] == 0
    assert all(
        gates[name] is None
        for name in ("probability_quality", "decision_benefit", "retention", "efficiency")
    )
    assert result["prd_gap_findings"]["verification"]["qualified"] is False
    assert result["prd_gap_findings"]["retained_learning"]["qualified"] is False
    assert result["prd_gap_findings"]["live_path_efficiency"]["qualified"] is False
    assert result["historical_boundaries"]["Exp7760"]["spec_mismatch"] is True
    assert result["historical_boundaries"]["Exp7759"]["canonical_parsed"] == 1
    assert result["historical_boundaries"]["Exp7759"]["alternate_parsed"] == 2
    assert result["MODEL_SPECS"] == result["model_specs"] == []
    assert result["model_invocation_counts"]["loads"] == 0
    assert result["source_artifact_hashes"]
    assert len(result["field_principles"]) >= 25
    assert capstone.cold_replay(result, ROOT) == []


def test_owned_validation_failure_and_contract_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7780-GATES: own failure disqualifies; wrong contract fails."""
    receipt = {
        "name": "ruff_format",
        "passed": False,
        "exit_code": 1,
        "log_path": "/tmp/fmt.log",
        "log_sha256": "sha256:test",
    }
    result = capstone.build_artifact(ROOT, {}, [receipt])
    assert result["verdict_class"] == "disqualified"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert any(
        f["field"] == "validation.ruff_format.exit_code" for f in result["gate_check_summary"]
    )
    source = capstone.authority(ROOT)
    source["comparison"] = {"passed": False, "errors": ["row_mismatch"]}
    monkeypatch.setattr(capstone, "authority", lambda _root: source)
    changed = capstone.build_artifact(ROOT, {}, [{"passed": True}])
    assert changed["verdict_class"] == "disqualified"
    assert any(f["field"] == "table_json_yaml_match" for f in changed["gate_check_summary"])


def test_cold_replay_detects_row_and_hash_changes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7780-REPLAY: independent bytes bind each row."""
    result = capstone.build_artifact(ROOT, {}, [{"passed": True}])
    altered = deepcopy(result)
    altered["rows"][0]["producer_hash"] = "sha256:wrong"
    assert "rows" in capstone.cold_replay(altered, ROOT)
    altered = deepcopy(result)
    altered["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
    assert "source_artifact_hashes" in capstone.cold_replay(altered, ROOT)
    fixture = tmp_path / "candidate.json"
    fixture.write_text(json.dumps(result))
    assert json.loads(fixture.read_text())["experiment_id"] == 7780


def test_reuse_exact_repository_diagnostic(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7780-REPLAY: a logged full-suite failure runs once."""
    cli = load_cli()
    raw = tmp_path / "raw"
    raw.mkdir()
    log = raw / "full.log"
    log.write_text("18 collection errors\n")
    receipt = {
        "name": "full_python_suite",
        "command_argv": ["pytest", "tests/python"],
        "log_path": str(log),
        "log_sha256": cli.sha256_file(log),
        "exit_code": 2,
        "passed": False,
    }
    (raw / "terminal_candidate.json").write_text(
        json.dumps({"validation_receipts": {"repository_collection": [receipt]}})
    )
    assert cli.existing_repository_receipt(tmp_path, raw) == [receipt]
    (raw / "terminal_candidate.json").write_text(
        json.dumps({"validation_receipts": {"repository_collection": []}})
    )
    assert cli.existing_repository_receipt(tmp_path, raw) is None
    (raw / "terminal_candidate.json").write_text(
        json.dumps({"validation_receipts": {"repository_collection": [receipt]}})
    )
    log.write_text("changed\n")
    assert cli.existing_repository_receipt(tmp_path, raw) is None


def test_cli_readers_and_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7780-REPLAY: CLI can reject changed rows and parse arguments."""
    cli = load_cli()
    assert cli.span("one", 1.0, 2.0, 3)["duration_s"] == 1.0
    raw = tmp_path / "raw"
    raw.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": [{"unit_id": "x"}]}))
    monkeypatch.setattr(cli, "RAW", Path("raw"))
    monkeypatch.setattr(cli, "cold_replay", lambda value, root: [])
    assert cli.read_candidate(candidate, tmp_path) == ["raw_rows"]
    (raw / "rows.json").write_text(json.dumps([{"unit_id": "x"}]))
    assert cli.read_candidate(candidate, tmp_path) == []
    (raw / "rows.json").write_text("[]")
    assert cli.read_candidate(candidate, tmp_path) == ["raw_rows"]
    monkeypatch.setattr(cli, "read_candidate", lambda path, root: [])
    assert cli.main(["--cold-validate", str(candidate)]) == 0
    monkeypatch.setattr(cli, "read_candidate", lambda path, root: ["rows"])
    assert cli.main(["--cold-validate", str(candidate)]) == 1
    calls = []
    monkeypatch.setattr(cli, "run_experiment", lambda *args: calls.append(args))
    assert cli.main(["--root", str(tmp_path), "--date", "20260927"]) == 0
    assert calls
    monkeypatch.setattr(sys, "argv", [str(ROOT / cli.CLI), "--help"])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(ROOT / cli.CLI), run_name="__main__")
    assert done.value.code == 0


def test_cli_publication_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7780-GATES: FoVer gate is recorded as its own child."""
    cli = load_cli()

    class Completed:
        stdout = '{"G1": true, "G2": false, "G3": true, "G4": true, "paper_ready": false, "unmet_gates": ["G2"]}'
        returncode = 0

    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: Completed())
    gate = cli.publication(0.0)
    assert gate["paper_ready"] is False
    assert gate["command_exit"] == 0
    assert gate["result_hash"]


def test_cli_orchestration_and_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7780-GATES: exact candidate failures close own readiness."""
    cli = load_cli()
    monkeypatch.setattr(cli, "RAW", Path("raw"))
    raw = tmp_path / "raw"
    raw.mkdir()
    scope = {
        "changed_modules": [str(cli.MODULE)],
        "direct_tests": [str(cli.TEST)],
        "transitive_consumer_tests": [],
        "static_paths": [str(cli.CLI)],
    }
    (raw / "frozen_affected_scope.json").write_text(json.dumps(scope))
    monkeypatch.setattr(cli, "authority", lambda root: {"tasks": [{}] * 14})
    monkeypatch.setattr(cli, "account", lambda root, tasks: ([{}] * 14, [], []))
    monkeypatch.setattr(cli, "publication", lambda start: {"G1": True, "paper_ready": False})
    private_count = [0]

    def private_dir(**kwargs):
        private_count[0] += 1
        return str(tmp_path / f"private-{private_count[0]}")

    monkeypatch.setattr(cli.tempfile, "mkdtemp", private_dir)
    monkeypatch.setattr(
        cli,
        "build_scoped_commands",
        lambda *args, **kwargs: [cli.CommandSpec("focus", ("python",), "affected")],
    )

    terminal_fails = False
    seen: list[str] = []

    def fake_commands(root, commands, *, log_dir, **kwargs):
        log_dir.mkdir(parents=True, exist_ok=True)
        found = []
        for item in commands:
            seen.append(item.name)
            log = log_dir / f"{item.name}.log"
            log.write_text(f"{item.name}\n")
            passed = not (terminal_fails and item.name == "adversarial_verify")
            found.append(
                {
                    "name": item.name,
                    "command_argv": list(item.argv),
                    "log_path": str(log),
                    "log_sha256": cli.sha256_file(log),
                    "exit_code": 0 if passed else 1,
                    "passed": passed,
                }
            )
        return found

    monkeypatch.setattr(cli, "run_commands", fake_commands)

    def fake_artifact(root, pub, receipts, spans, duration):
        return {
            "rows": [{"unit_id": "exp7780-capstone"}],
            "validation_receipts": {},
            "gate_check_summary": [],
            "acceptance_gate_results": {"validity": True, "readiness": 1},
            "task_dispositions": [{"task_id": "exp7780-capstone"}],
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_required_v676_evidence",
            "capstone_complete_score": 0,
        }

    monkeypatch.setattr(cli, "build_artifact", fake_artifact)
    with pytest.raises(ValueError, match="run date"):
        cli.run_experiment(tmp_path, "20260926", Path("out.json"))
    frozen = raw / "frozen_affected_scope.json"
    frozen.unlink()
    with pytest.raises(FileNotFoundError):
        cli.run_experiment(tmp_path, "20260927", Path("out.json"))
    frozen.write_text(json.dumps({**scope, "direct_tests": []}))
    with pytest.raises(ValueError, match="scope mismatch"):
        cli.run_experiment(tmp_path, "20260927", Path("out.json"))
    frozen.write_text(json.dumps(scope))

    first = cli.run_experiment(tmp_path, "20260927", Path("out.json"))
    assert first["validation_receipts"]["repository_collection_healthy"] is True
    assert "full_python_suite" in seen
    seen.clear()
    terminal_fails = True
    second = cli.run_experiment(tmp_path, "20260927", Path("out.json"))
    assert second["verdict_class"] == "disqualified"
    assert second["flagged_adversarial"] is True
    assert second["acceptance_gate_results"]["readiness"] == 0
    assert second["rows"][-1]["verdict_class"] == "disqualified"
    assert "full_python_suite" not in seen
    assert json.loads((tmp_path / "out.json").read_text())["flagged_adversarial"] is True
