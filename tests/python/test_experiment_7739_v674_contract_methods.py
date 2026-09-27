"""REQ-REPORT-7739 and REQ-HARNESS-7739 custody checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from carnot import experiment_7739_v674_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]
DESIGN = (ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text()
ROADMAP = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text())


def test_three_sources_and_private_mutations() -> None:
    """SCENARIO-REPORT-7739-CONTRACT: all fourteen rows and fields matter."""

    result = subject.compare_contract(DESIGN, ROADMAP)
    assert result["passed"]
    assert len(result["rows"]) == 14
    assert all(row["matched"] and row["raw_denominator"] > 0 for row in result["rows"])
    for name in ("delete", "reorder", "milestone", "path", "model", "gate_field"):
        changed = subject.mutate_roadmap(ROADMAP, name)
        assert changed != ROADMAP
        assert not subject.compare_contract(DESIGN, changed)["passed"], name


def test_independent_design_sources_fail_closed() -> None:
    """SCENARIO-REPORT-7739-CONTRACT: no authority is reconstructed from YAML."""

    assert not subject.compare_contract(
        DESIGN.replace("## Exact Task Contract", "## Removed"), ROADMAP
    )["passed"]
    assert not subject.compare_contract(DESIGN.replace("```json", "```removed", 1), ROADMAP)[
        "passed"
    ]
    changed = deepcopy(ROADMAP)
    changed["tasks"][2]["gated_on"][0]["artifact_field"] = "typo_ready_score"
    assert not subject.compare_contract(DESIGN, changed)["passed"]
    changed = deepcopy(ROADMAP)
    changed["tasks"][0].pop("MODEL_SPECS")
    assert not subject.compare_contract(DESIGN, changed)["passed"]


def test_authority_selection(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7739-SELECTED: only V674 authority can qualify."""

    active = tmp_path / "research-roadmap.yaml"
    active.write_text(yaml.safe_dump(ROADMAP))
    assert subject.resolve_authority(tmp_path)[0] == active
    staged = tmp_path / "research-roadmap-next.yaml"
    staged.write_text(yaml.safe_dump(ROADMAP))
    assert subject.resolve_authority(tmp_path)[0] == staged
    stale = deepcopy(ROADMAP)
    stale["milestone"] = "2026.09.673"
    staged.write_text(yaml.safe_dump(stale))
    assert subject.resolve_authority(tmp_path)[0] == active
    active.unlink()
    with pytest.raises(ValueError, match="matching V674 authority"):
        subject.resolve_authority(tmp_path)


def test_raw_replay_detects_row_and_input_changes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7739-TERMINAL: independent replay binds bytes."""

    result = subject.compare_contract(DESIGN, ROADMAP)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(result["rows"]))
    assert subject.reduce_raw(raw, DESIGN, ROADMAP) == result
    altered = deepcopy(result["rows"])
    altered[0]["matched"] = False
    raw.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="raw rows differ"):
        subject.reduce_raw(raw, DESIGN, ROADMAP)
    raw.write_text(json.dumps(result["rows"]))
    with pytest.raises(ValueError, match="raw rows differ"):
        subject.reduce_raw(raw, DESIGN + "\nchanged", ROADMAP)


def test_blocked_operand_and_old_custody() -> None:
    """SCENARIO-REPORT-7739-CUSTODY: old disqualification stays historical."""

    check = subject.operand("path", "exp7726", "missing", "exists", True, False)
    assert check == {
        "check": "path",
        "upstream": "exp7726",
        "artifact_path": "missing",
        "field": "exists",
        "op": "==",
        "expected": True,
        "observed": False,
        "passed": False,
    }
    inventory = subject.input_inventory(ROOT, ROOT / "research-roadmap.yaml")
    old = next(
        row
        for row in inventory
        if row["path"].endswith("experiment_7726_v673_contract_methods.json")
    )
    assert old["role"] == "historical_disqualified"
    assert old["exists"]
    assert (ROOT / "python/carnot/models/gibbs/__init__.py").is_file()
    assert not (ROOT / "python/carnot/models/gibbs.py").is_file()


def test_artifact_cold_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7739-TERMINAL: summary cannot hide changed bytes."""

    comparison = subject.compare_contract(DESIGN, ROADMAP)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(comparison["rows"]))
    value = subject.build_artifact(ROOT, ROOT / "research-roadmap.yaml", comparison, raw, [], [])
    assert subject.validate_candidate(value, ROOT, raw)
    assert value["verdict_class"] == "disqualified"
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    assert value["MODEL_SPECS"] == value["planned_MODEL_SPECS"] == []
    assert value["model_invoked"] is False
    assert value["contract_ready_score"] == 0
    damaged = deepcopy(value)
    damaged["rows"][0]["matched"] = False
    damaged["reproducibility_checksum"] = subject.checksum(damaged)
    assert not subject.validate_candidate(damaged, ROOT, raw)


def test_malformed_design_and_unknown_mutation() -> None:
    """SCENARIO-REPORT-7739-CONTRACT: malformed table and machine fail closed."""

    assert (
        "design_table_invalid"
        in subject.parse_design(DESIGN.replace("| [] | [] |", "| broken | [] |", 1))[2]
    )
    assert (
        "design_json_invalid"
        in subject.parse_design(
            DESIGN.replace('"milestone": "2026.09.674"', '"milestone": "stale"', 1)
        )[2]
    )
    assert (
        "design_json_invalid"
        in subject.parse_design(DESIGN.replace('"milestone": "2026.09.674"', '"milestone": ', 1))[2]
    )
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate_roadmap(ROADMAP, "unknown")


def test_candidate_rejects_missing_or_changed_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7739-CUSTODY: a changed hash cannot be self-certified."""

    comparison = subject.compare_contract(DESIGN, ROADMAP)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(comparison["rows"]))
    value = subject.build_artifact(ROOT, ROOT / "research-roadmap.yaml", comparison, raw, [], [])
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "wrong"
    assert not subject.validate_candidate(changed, ROOT, raw)
    changed = deepcopy(value)
    next(
        row
        for row in changed["source_artifact_hashes"]
        if row["role"] == "planned_output_not_input"
    )["sha256"] = "wrong"
    changed["reproducibility_checksum"] = subject.checksum(changed)
    assert not subject.validate_candidate(changed, ROOT, raw)
    changed = deepcopy(value)
    next(row for row in changed["source_artifact_hashes"] if row["role"] == "current_input")[
        "exists"
    ] = False
    changed["reproducibility_checksum"] = subject.checksum(changed)
    assert not subject.validate_candidate(changed, ROOT, raw)
    assert not subject.validate_candidate(value, ROOT, tmp_path / "missing.json")
    raw.write_text("bad json")
    assert not subject.validate_candidate(value, ROOT, raw)


def test_blocked_and_ready_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7739-CUSTODY: absence blocks; passing checks only qualify planning."""

    comparison = subject.compare_contract(DESIGN, ROADMAP)
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(comparison["rows"]))
    good = [{"name": "focused_pytest", "passed": True}]
    value = subject.build_artifact(ROOT, ROOT / "research-roadmap.yaml", comparison, raw, good, [])
    assert value["verdict_class"] == "null"
    assert value["contract_ready_score"] == 1
    assert value["acceptance_gate_results"]["readiness"] is None
    flagged = subject.build_artifact(
        ROOT,
        ROOT / "research-roadmap.yaml",
        comparison,
        raw,
        [{"name": "adversarial_verify", "passed": False}],
        [],
    )
    assert flagged["flagged_adversarial"] is True
    assert flagged["verdict_class"] == "disqualified"
    monkeypatch.setattr(subject, "METHOD", tmp_path / "missing-method.md")
    blocked = subject.build_artifact(
        ROOT, ROOT / "research-roadmap.yaml", comparison, raw, good, []
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["artifact_path"] == str(subject.METHOD)


def test_plan_and_wrong_milestone(tmp_path: Path) -> None:
    """SCENARIO-HARNESS-7739-SELECTED: scope and authority stay explicit."""

    plan = subject.validation_plan(ROOT, ROOT / "research-roadmap.yaml", tmp_path)
    names = [item.name for item in plan]
    assert "changed_module_coverage_report" in names
    assert "roadmap_schema" in names
    assert "prior_failure" in names
    assert "exclusion_manifest" in names
    assert "roadmap_gate_audit" in names
    assert "harness_fit" in names
    assert "arc_floor" in names
    assert "overdue_priority" in names
    assert "prompt_path" in names
    wrong = deepcopy(ROADMAP)
    wrong["milestone"] = "2026.09.673"
    assert "roadmap_milestone" in subject.compare_contract(DESIGN, wrong)["errors"]


@pytest.mark.parametrize("terminal_failure", [False, True])
def test_private_run_and_cli_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_failure: bool
) -> None:
    """SCENARIO-REPORT-7739-TERMINAL: fixture execution replays in a new CLI process."""

    authority = tmp_path / "research-roadmap.yaml"
    authority.write_text(yaml.safe_dump(ROADMAP))
    for row in subject.input_inventory(ROOT, ROOT / "research-roadmap.yaml"):
        if row["role"] in {"planned_output_not_input"} or row["path"] == "research-roadmap.yaml":
            continue
        target = tmp_path / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / row["path"], target)
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    fake = subject.CommandSpec("fake_validation", ("true",), "fixture")
    monkeypatch.setattr(subject, "validation_plan", lambda *_: [fake])

    def fake_run(_root: Path, commands: list[subject.CommandSpec], **_kw: object) -> list[dict]:
        return [
            {
                "name": item.name,
                "passed": not (terminal_failure and item.name == "adversarial_verify"),
                "exit_code": int(terminal_failure and item.name == "adversarial_verify"),
                "command_argv": list(item.argv),
                "log_sha256": "sha256:fixture",
            }
            for item in commands
        ]

    monkeypatch.setattr(subject, "run_commands", fake_run)
    value = subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
    assert (tmp_path / subject.RESULT).is_file()
    assert value["contract_ready_score"] == int(not terminal_failure)
    raw = tmp_path / subject.RAW / "rows.json"
    candidate = tmp_path / subject.RESULT
    assert subject.main(["--cold-validate", str(candidate), "--raw", str(raw)]) == 0
    replay = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / subject.CLI),
            "--cold-validate",
            str(candidate),
            "--raw",
            str(raw),
            "--fixture-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert json.loads(replay.stdout)["rows"] == 14
    bad = deepcopy(value)
    bad["rows"][0]["matched"] = False
    bad["reproducibility_checksum"] = subject.checksum(bad)
    candidate.write_text(json.dumps(bad))
    assert subject.main(["--cold-validate", str(candidate), "--raw", str(raw)]) == 1
    assert subject.main(["--cold-validate", str(tmp_path / "absent.json"), "--raw", str(raw)]) == 1
    if not terminal_failure:
        with pytest.raises(ValueError, match="root, date"):
            subject.run_experiment(tmp_path, "20260926", subject.RESULT)
        original_mutate = subject.mutate_roadmap
        monkeypatch.setattr(subject, "mutate_roadmap", lambda roadmap, _name: roadmap)
        with pytest.raises(RuntimeError, match="private mutation"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
        monkeypatch.setattr(subject, "mutate_roadmap", original_mutate)
        original_validate = subject.validate_candidate
        monkeypatch.setattr(subject, "validate_candidate", lambda *_: False)
        with pytest.raises(RuntimeError, match="candidate failed"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
        calls = iter([True, False])
        monkeypatch.setattr(subject, "validate_candidate", lambda *_: next(calls))
        with pytest.raises(RuntimeError, match="terminal candidate failed"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
        monkeypatch.setattr(subject, "validate_candidate", original_validate)
        count = 0

        def mismatched(
            _root: Path, commands: list[subject.CommandSpec], **_kw: object
        ) -> list[dict]:
            nonlocal count
            count += 1
            rows = fake_run(_root, commands)
            if count == 3:
                rows[0]["passed"] = False
            return rows

        monkeypatch.setattr(subject, "run_commands", mismatched)
        with pytest.raises(RuntimeError, match="exact terminal reader"):
            subject.run_experiment(tmp_path, subject.RUN_DATE, subject.RESULT)
        monkeypatch.setattr(subject, "run_experiment", lambda *_: {})
        assert subject.main([]) == 0
    damaged = deepcopy(value)
    next(row for row in damaged["source_artifact_hashes"] if row["role"] == "current_input")[
        "sha256"
    ] = "wrong"
    damaged["reproducibility_checksum"] = subject.checksum(damaged)
    assert not subject.validate_candidate(damaged, ROOT, raw)
