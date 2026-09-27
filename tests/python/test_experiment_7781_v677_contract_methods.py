"""REQ-REPORT-7781: V677 authorities, custody and replay."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from carnot import experiment_7781_v677_contract_methods as subject


def load_cli():
    """Load the real entrypoint without starting its command line mode."""
    spec = importlib.util.spec_from_file_location("exp7781_cli", subject.ROOT / subject.CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def authorities(tmp_path: Path) -> tuple[Path, str, dict]:
    """Use immutable V677 bytes so a later active roadmap cannot rewrite history."""
    design = (subject.ROOT / subject.DESIGN_SNAPSHOT).read_text()
    roadmap = yaml.safe_load((subject.ROOT / subject.YAML_SNAPSHOT).read_text())
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    return tmp_path, design, roadmap


def test_matching_authority_and_exact_rows(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7781-CONTRACT: fourteen rows need three agreeing sources."""
    root, design, roadmap = authorities
    selected, actual, candidates = subject.resolve_authority(root)
    assert selected.name == "research-roadmap.yaml" and actual == roadmap
    assert candidates[0]["exists"] is False
    (root / "research-roadmap-next.yaml").write_text(yaml.safe_dump(roadmap))
    assert subject.resolve_authority(root)[0].name == "research-roadmap-next.yaml"
    result = subject.compare_contract(design, roadmap)
    assert result["passed"] and len(result["rows"]) == 14
    assert [r["unit_id"].split("-", 1)[0] for r in result["rows"]] == [
        f"exp{i}" for i in range(7781, 7795)
    ]
    assert all(r["matched"] for r in result["rows"])
    assert roadmap["tasks"][0].get("gated_on", []) == []
    roadmap["milestone"] = "2026.09.678"
    (root / "research-roadmap-next.yaml").write_text(yaml.safe_dump(roadmap))
    (root / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    with pytest.raises(ValueError, match="V677 authority"):
        subject.resolve_authority(root)


@pytest.mark.parametrize(
    "name",
    [
        "drop",
        "reorder",
        "title",
        "unknown_producer",
        "gate_field",
        "model",
        "substrate",
        "prior_experiment_id",
        "prior_verdict",
        "prior_addressed_by",
        "prior_retirement",
    ],
)
def test_private_mutations_rejected(authorities: tuple[Path, str, dict], name: str) -> None:
    """SCENARIO-REPORT-7781-CONTRACT: each changed operand fails closed."""
    _, design, roadmap = authorities
    assert not subject.compare_contract(design, subject.mutate(roadmap, name))["passed"]


def test_design_and_gate_fields_are_independent(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7781-CONTRACT: table and producer prompt are checked."""
    _, design, roadmap = authorities
    altered = design.replace(
        "Bind fourteen tasks and register causal evidence methods |", "Wrong title |", 1
    )
    assert not subject.compare_contract(altered, roadmap)["passed"]
    altered = deepcopy(roadmap)
    altered["tasks"][1]["prompt"] = altered["tasks"][1]["prompt"].replace(
        "historical_fixture_ready_score", "wrong", 1
    )
    assert not subject.compare_contract(design, altered)["passed"]
    with pytest.raises(ValueError, match="JSON contract missing"):
        subject.parse_design(design.split("<!-- V677_TASK_CONTRACT_START -->")[0])
    with pytest.raises(ValueError, match="JSON milestone"):
        subject.parse_design(
            design.replace('"milestone": "2026.09.677"', '"milestone": "wrong"', 1)
        )


def test_v676_custody_and_failed_operands(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7781-CUSTODY: missing producers stay distinct from receipts."""
    _, design, roadmap = authorities
    rows = subject.prior_inventory(subject.ROOT)
    assert len(rows) == 14
    assert sum(r["producer_state"] == "missing" for r in rows) == 6
    assert sum(r["producer_state"] == "disqualified" for r in rows) == 6
    assert sum(r["producer_state"] == "blocked" for r in rows) == 1
    assert sum(r["producer_state"] == "circular_positive" for r in rows) == 1
    assert all(r["producer_sha256"] is None for r in rows if r["producer_state"] == "missing")
    assert any(r["pre_gate_path"] for r in rows if r["producer_state"] == "missing")
    comparison = subject.compare_contract(design, subject.mutate(roadmap, "title"))
    failures = subject.failed_checks(
        comparison, [{"path": "missing", "exists": False, "sha256": None}]
    )
    assert any(r["field"] == "exists" and r["observed"] is False for r in failures)
    assert any(r["field"] == "title" for r in failures)


def test_cold_replay_survives_later_active_roadmap(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7781-TERMINAL: immutable snapshots own replay."""
    root, design, roadmap = authorities
    for relative, data in (
        (subject.DESIGN_SNAPSHOT, design),
        (subject.YAML_SNAPSHOT, yaml.safe_dump(roadmap)),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    rows = subject.compare_contract(design, roadmap)["rows"]
    raw = root / "rows.json"
    raw.write_text(json.dumps(rows))
    sources = subject.source_hashes(root, [subject.DESIGN_SNAPSHOT, subject.YAML_SNAPSHOT])
    assert subject.cold_validate(root, raw, sources)
    later = deepcopy(roadmap)
    later["milestone"] = "2026.09.678"
    (root / "research-roadmap.yaml").write_text(yaml.safe_dump(later))
    assert subject.cold_validate(root, raw, sources)
    rows[0]["matched"] = False
    raw.write_text(json.dumps(rows))
    assert not subject.cold_validate(root, raw, sources)
    (root / subject.DESIGN_SNAPSHOT).write_text("changed")
    assert not subject.cold_validate(root, raw, sources)


def test_malformed_milestone_and_history_fail_closed(
    authorities: tuple[Path, str, dict], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7781-CONTRACT/CUSTODY: reject stale authority and history."""
    _, design, roadmap = authorities
    changed = deepcopy(roadmap)
    changed["milestone"] = "2026.09.678"
    assert "roadmap_milestone" in subject.compare_contract(design, changed)["errors"]
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate(roadmap, "not_registered")
    path = tmp_path / subject.PRESERVED
    path.parent.mkdir(parents=True)
    path.write_text("no machine block")
    with pytest.raises(ValueError, match="machine contract missing"):
        subject.prior_inventory(tmp_path)
    preserved = (subject.ROOT / subject.PRESERVED).read_text()
    path.write_text(
        preserved.replace('"id": "exp7767-contract-methods"', '"id": "exp9999-contract-methods"', 1)
    )
    with pytest.raises(ValueError, match="task order changed"):
        subject.prior_inventory(tmp_path)


@pytest.mark.parametrize("fail_exact", [False, True])
def test_cli_freezes_candidate_before_publication(
    authorities: tuple[Path, str, dict], monkeypatch: pytest.MonkeyPatch, fail_exact: bool
) -> None:
    """SCENARIO-REPORT-7781-TERMINAL: terminal failure closes every readiness score."""
    root, design, roadmap = authorities
    cli = load_cli()
    monkeypatch.setattr(cli, "ROOT", root)
    for relative, content in (
        (subject.DESIGN, design),
        (subject.DESIGN_SNAPSHOT, design),
        (subject.YAML_SNAPSHOT, yaml.safe_dump(roadmap)),
        (subject.RAW / "frozen_affected_scope.json", "{}"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (root / "results").mkdir(exist_ok=True)
    monkeypatch.setattr(
        cli,
        "prior_inventory",
        lambda _root: [
            {
                "producer_path": f"results/experiment_{number}.json",
                "producer_state": "missing",
                "producer_sha256": None,
                "pre_gate_path": None,
            }
            for number in range(7767, 7781)
        ],
    )
    calls = []

    def fake_commands(_root, commands, *, log_dir, heartbeat_s):
        calls.append([spec.name for spec in commands])
        failed = fail_exact and len(calls) == 4
        return [
            {
                "name": spec.name,
                "scope": spec.scope,
                "passed": not failed,
                "exit_code": 1 if failed else 0,
                "log_path": str(log_dir / f"{spec.name}.log"),
                "log_sha256": "sha256:fixture",
            }
            for spec in commands
        ]

    monkeypatch.setattr(cli, "run_commands", fake_commands)
    with pytest.raises(ValueError, match="run date"):
        cli.run_experiment("wrong")
    snapshot_path = root / subject.YAML_SNAPSHOT
    clean_snapshot = snapshot_path.read_text()
    altered = deepcopy(roadmap)
    altered["tasks"][0]["title"] = "Drifted title"
    snapshot_path.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError, match="frozen snapshots"):
        cli.run_experiment("20260927")
    snapshot_path.write_text(clean_snapshot)
    original_mutate = cli.mutate
    monkeypatch.setattr(cli, "mutate", lambda value, _name: value)
    with pytest.raises(RuntimeError, match="private mutation"):
        cli.run_experiment("20260927")
    monkeypatch.setattr(cli, "mutate", original_mutate)
    original_verify = cli.verify_candidate
    monkeypatch.setattr(cli, "verify_candidate", lambda _path, _raw: False)
    with pytest.raises(RuntimeError, match="independent row reduction"):
        cli.run_experiment("20260927")
    monkeypatch.setattr(cli, "verify_candidate", original_verify)
    calls.clear()
    result = cli.run_experiment("20260927")
    assert (root / subject.RESULT).is_file()
    assert len(result["rows"]) == 14
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert result["verdict_class"] in {"blocked", "disqualified"}
    assert len(calls) == 4
    if fail_exact:
        assert result["honest_verdict"] == "complete_disqualified_v677_terminal_validation"
    assert (
        cli.main(
            [
                "--cold-validate",
                str(root / subject.RESULT),
                "--raw",
                str(root / subject.RAW / "rows.json"),
            ]
        )
        == 0
    )
    raw_path = root / subject.RAW / "rows.json"
    raw_path.write_text("[]")
    assert cli.main(["--cold-validate", str(root / subject.RESULT), "--raw", str(raw_path)]) == 1
    monkeypatch.setattr(cli, "run_experiment", lambda date: {"date": date})
    assert cli.main(["--date", "20260927"]) == 0


def test_cli_verdicts_and_script_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7781-TERMINAL: own failed validation disqualifies readiness."""
    cli = load_cli()
    comparison = subject.compare_contract(
        (subject.ROOT / subject.DESIGN_SNAPSHOT).read_text(),
        yaml.safe_load((subject.ROOT / subject.YAML_SNAPSHOT).read_text()),
    )
    sources = subject.source_hashes(subject.ROOT, [subject.DESIGN_SNAPSHOT, subject.YAML_SNAPSHOT])
    passed = {
        "name": "focused_pytest",
        "scope": "explicit_tests",
        "passed": True,
        "exit_code": 0,
        "log_path": "/tmp/ok.log",
        "log_sha256": "sha256:ok",
    }
    authority = subject.ROOT / "research-roadmap.yaml"
    ready = cli.build_artifact(authority, comparison, [passed], [], [], sources)
    assert ready["verdict_class"] == "circular_positive"
    assert ready["contract_ready_score"] == 1
    failed = dict(passed, passed=False, exit_code=1)
    rejected = cli.build_artifact(authority, comparison, [failed], [], [], sources)
    assert rejected["verdict_class"] == "disqualified"
    assert rejected["contract_ready_score"] == 0
    assert any(row["upstream_id"] == "focused_pytest" for row in rejected["gate_check_summary"])
    candidate = tmp_path / "invalid.json"
    candidate.write_text(json.dumps({"source_artifact_hashes": []}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(subject.CLI),
            "--cold-validate",
            str(candidate),
            "--raw",
            str(tmp_path / "unused.json"),
        ],
    )
    with pytest.raises(SystemExit, match="1"):
        runpy.run_path(str(subject.ROOT / subject.CLI), run_name="__main__")
