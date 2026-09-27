"""REQ-REPORT-7753: exercise the V675 contract on private copies."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from carnot import experiment_7753_v675_contract_methods as subject


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, str, dict]:
    """Keep every mutated authority outside the public repository."""
    design = (subject.ROOT / subject.DESIGN).read_text()
    roadmap = yaml.safe_load((subject.ROOT / "research-roadmap.yaml").read_text())
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    return tmp_path, design, roadmap


def test_authority_uses_matching_milestone_only(inputs: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7753-CONTRACT: staged authority wins only when matching."""
    root, _, roadmap = inputs
    selected, value, candidates = subject.resolve_authority(root)
    assert selected.name == "research-roadmap.yaml" and value == roadmap
    assert candidates[0]["exists"] is False
    (root / "research-roadmap-next.yaml").write_text(yaml.safe_dump(roadmap))
    assert subject.resolve_authority(root)[0].name == "research-roadmap-next.yaml"
    roadmap["milestone"] = "2026.09.676"
    (root / "research-roadmap-next.yaml").write_text(yaml.safe_dump(roadmap))
    (root / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    with pytest.raises(ValueError, match="matching V675"):
        subject.resolve_authority(root)


def test_three_authorities_match(inputs: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7753-CONTRACT: no source fills another's missing field."""
    _, design, roadmap = inputs
    result = subject.compare_contract(design, roadmap)
    assert result["passed"] and len(result["rows"]) == 14
    assert all(row["matched"] for row in result["rows"])
    assert [row["unit_id"] for row in result["rows"]] == [
        f"exp{i}-{roadmap['tasks'][i - 7753]['id'].split('-', 1)[1]}" for i in range(7753, 7767)
    ]


@pytest.mark.parametrize(
    "mutation",
    ["delete", "reorder", "title", "producer_field", "substrate", "prior_field", "retirement"],
)
def test_private_mutations_fail(inputs: tuple[Path, str, dict], mutation: str) -> None:
    """SCENARIO-REPORT-7753-CONTRACT: every registered drift fails closed."""
    _, design, roadmap = inputs
    changed = deepcopy(roadmap)
    assert not subject.compare_contract(design, subject.mutate(changed, mutation))["passed"]


def test_table_is_separately_parsed(inputs: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7753-CONTRACT: a damaged table cannot borrow JSON."""
    _, design, roadmap = inputs
    damaged = design.replace(
        "Bind fourteen tasks and register evidence-view methods |", "Wrong title |", 1
    )
    assert not subject.compare_contract(damaged, roadmap)["passed"]


def test_v674_missing_producers_and_receipts_are_distinct(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7753-CUSTODY: a queue receipt is not a scientific result."""
    design = (subject.ROOT / subject.PRESERVED).read_text()
    rows = subject.v674_inventory(subject.ROOT, design)
    missing = [row for row in rows if row["producer_state"] == "missing"]
    assert len(missing) == 6
    assert {7741, 7749} <= {row["experiment"] for row in missing if row["pre_gate_path"]}
    assert all(row["producer_sha256"] is None for row in missing)
    assert any(row["producer_state"] == "measured_null" for row in rows)
    assert any(row["producer_state"] == "disqualified" for row in rows)
    assert tmp_path.is_dir()


def test_raw_rows_reduced_against_authorities(inputs: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7753-TERMINAL: cold reduction detects changed raw rows."""
    root, design, roadmap = inputs
    rows = subject.compare_contract(design, roadmap)["rows"]
    raw = root / "raw.json"
    raw.write_text(json.dumps(rows))
    assert subject.cold_reduce(raw, design, roadmap)["passed"]
    rows[0]["matched"] = False
    raw.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="raw rows"):
        subject.cold_reduce(raw, design, roadmap)


def test_artifact_and_cold_custody(
    inputs: tuple[Path, str, dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7753-CUSTODY/TERMINAL: readiness needs passing readers."""
    root, design, roadmap = inputs
    for path in (subject.DESIGN, subject.PRESERVED, subject.RAW / "frozen_affected_scope.json"):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        source = subject.ROOT / path
        shutil.copyfile(source, target)
    raw = root / subject.RAW / "rows.json"
    comparison = subject.compare_contract(design, roadmap)
    raw.write_text(json.dumps(comparison["rows"]))
    authority = root / "research-roadmap.yaml"
    candidates = [{"path": "research-roadmap.yaml", "exists": True, "milestone": subject.MILESTONE}]
    inventory = [
        {
            "path": str(subject.DESIGN),
            "role": "current_input",
            "exists": True,
            "sha256": subject.sha256_file(root / subject.DESIGN),
            "date": "2026-09-27",
            "imported_fields": ["bytes"],
            "eligible": True,
        }
    ]
    monkeypatch.setattr(subject, "input_inventory", lambda *_: inventory)
    span = {"phase": "fixture", "duration_s": 0.1, "completed_units": 14}
    passed = [{"name": "adversarial_verify", "passed": True}]
    value = subject.build_artifact(root, authority, comparison, raw, passed, [span], candidates)
    assert value["contract_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert subject.cold_validate(value, root, raw)
    changed_rows = deepcopy(value)
    changed_rows["rows"] = []
    changed_rows["reproducibility_checksum"] = subject.checksum(changed_rows)
    assert not subject.cold_validate(changed_rows, root, raw)
    extra = root / "extra.txt"
    extra.write_text("before")
    value["source_artifact_hashes"].append(
        {"path": "extra.txt", "role": "current_input", "sha256": subject.sha256_file(extra)}
    )
    value["reproducibility_checksum"] = subject.checksum(value)
    extra.write_text("after")
    assert not subject.cold_validate(value, root, raw)
    extra.write_text("before")
    assert subject.cold_validate(value, root, raw)
    failed = subject.build_artifact(
        root,
        authority,
        comparison,
        raw,
        [{"name": "adversarial_verify", "passed": False}],
        [span],
        candidates,
    )
    assert failed["verdict_class"] == "disqualified" and failed["flagged_adversarial"]
    assert value["model_invocation_counts"]["loads"] == 0
    assert len(value["source_artifact_categories"]["missing_scientific_producers"]) == 14
    value["reproducibility_checksum"] = "wrong"
    assert not subject.cold_validate(value, root, raw)
    value["reproducibility_checksum"] = subject.checksum(value)
    (root / subject.DESIGN).write_text("changed")
    assert not subject.cold_validate(value, root, raw)
    (root / subject.DESIGN).write_text(design)
    assert subject.cold_validate(value, root, raw)
    raw.write_text("[]")
    assert not subject.cold_validate(value, root, raw)
    raw.write_text(json.dumps(comparison["rows"]))
    (root / subject.DESIGN).unlink()
    assert not subject.cold_validate(value, root, raw)


def test_blocked_and_disqualified_builder(inputs: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7753-TERMINAL: absent input and failed check remain distinct."""
    root, design, roadmap = inputs
    target = root / subject.PRESERVED
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(subject.ROOT / subject.PRESERVED, target)
    scope = root / subject.RAW / "frozen_affected_scope.json"
    scope.parent.mkdir(parents=True, exist_ok=True)
    scope.write_text("{}")
    raw = scope.parent / "rows.json"
    raw.write_text("[]")
    comparison = subject.compare_contract(design, roadmap)
    authority = root / "research-roadmap.yaml"
    inventory = subject.input_inventory(root, authority)
    assert any(not row["exists"] for row in inventory if row["role"] == "current_input")
    assert subject.failed_checks(comparison, inventory)
    value = subject.build_artifact(root, authority, comparison, raw, [], [], [])
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert not value["flagged_adversarial"]
    assert not subject.cold_validate(value, root, raw)


def test_defensive_parser_paths(inputs: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7753-CONTRACT: malformed independent sources fail closed."""
    _, design, roadmap = inputs
    assert subject.parse_design("missing")[2] == ["design_section_missing"]
    assert (
        "design_json_missing"
        in subject.parse_design("## Exact Task Contract\n| 1 | a | 1 | x | y | z | [] | [] |")[-1]
    )
    assert (
        "design_json_invalid"
        in subject.parse_design("## Exact Task Contract\n```json\n{}\n```")[-1]
    )
    bad = design.replace("| 1 | exp7753", "| 1 | exp7753", 1).replace(
        "| aggregation | [] | [] |", "| aggregation | invalid | [] |", 1
    )
    assert "design_table_invalid" in subject.parse_design(bad)[-1]
    wrong_milestone = deepcopy(roadmap)
    wrong_milestone["milestone"] = "2026.09.676"
    assert "roadmap_milestone" in subject.compare_contract(design, wrong_milestone)["errors"]
    mismatched = subject.compare_contract(design, subject.mutate(roadmap, "title"))
    failures = subject.failed_checks(
        mismatched,
        [{"path": str(subject.DESIGN), "role": "current_input", "exists": True, "sha256": "abc"}],
    )
    assert any(item["field"] == "title" for item in failures)
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate(roadmap, "unknown")
    with pytest.raises(ValueError, match="V674 machine"):
        subject.v674_inventory(Path("/tmp"), "## Exact Task Contract")
    broken = (
        (subject.ROOT / subject.PRESERVED)
        .read_text()
        .replace('"id":"exp7739-contract-methods"', '"id":"exp9999-contract-methods"', 1)
    )
    with pytest.raises(ValueError, match="historical task sequence"):
        subject.v674_inventory(subject.ROOT, broken)
    assert hashlib.sha256(design.encode()).hexdigest()


def test_nested_basetemp_parent_with_real_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7753-TERMINAL: child pytest can create nested basetemp."""
    child_test = tmp_path / "test_child.py"
    child_test.write_text("def test_live_child():\n    assert True\n")
    basetemp = tmp_path / "nested" / "parent" / "child"
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={basetemp}",
            str(child_test),
            "-q",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
