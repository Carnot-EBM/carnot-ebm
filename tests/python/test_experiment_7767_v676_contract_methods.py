"""REQ-REPORT-7767: current V676 contract and custody checks."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot import experiment_7767_v676_contract_methods as subject


@pytest.fixture
def authorities(tmp_path: Path) -> tuple[Path, str, dict]:
    """Use private copies so mutation tests cannot alter the active roadmap."""
    design = (subject.ROOT / subject.DESIGN).read_text()
    roadmap = yaml.safe_load((subject.ROOT / "research-roadmap.yaml").read_text())
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    return tmp_path, design, roadmap


def test_matching_authority_only(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7767-CONTRACT: a staged file needs the right milestone."""
    root, _, roadmap = authorities
    path, value, candidates = subject.resolve_authority(root)
    assert path.name == "research-roadmap.yaml" and value == roadmap
    assert candidates[0]["exists"] is False
    (root / "research-roadmap-next.yaml").write_text(yaml.safe_dump(roadmap))
    assert subject.resolve_authority(root)[0].name == "research-roadmap-next.yaml"
    roadmap["milestone"] = "2026.09.675"
    (root / "research-roadmap-next.yaml").write_text(yaml.safe_dump(roadmap))
    (root / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    with pytest.raises(ValueError, match="V676 authority"):
        subject.resolve_authority(root)


def test_three_source_contract(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7767-CONTRACT: all fourteen rows have independent agreement."""
    _, design, roadmap = authorities
    result = subject.compare_contract(design, roadmap)
    assert result["passed"] and len(result["rows"]) == 14
    assert [r["unit_id"].split("-", 1)[0] for r in result["rows"]] == [
        f"exp{i}" for i in range(7767, 7781)
    ]
    assert all(r["matched"] for r in result["rows"])
    assert not roadmap["tasks"][0].get("gated_on")


@pytest.mark.parametrize(
    "name",
    [
        "drop",
        "reorder",
        "title",
        "unknown_producer",
        "gate_field",
        "substrate",
        "prior_experiment_id",
        "prior_verdict",
        "prior_addressed_by",
        "prior_retirement",
    ],
)
def test_private_mutations_rejected(authorities: tuple[Path, str, dict], name: str) -> None:
    """SCENARIO-REPORT-7767-CONTRACT: every contractual field fails closed."""
    _, design, roadmap = authorities
    assert not subject.compare_contract(design, subject.mutate(deepcopy(roadmap), name))["passed"]


def test_table_and_gate_field_authority(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7767-CONTRACT: the table and producer prompt stand alone."""
    _, design, roadmap = authorities
    changed = design.replace(
        "Bind fourteen tasks and freeze evidence and validation contracts |",
        "Wrong title |",
        1,
    )
    assert not subject.compare_contract(changed, roadmap)["passed"]
    changed = deepcopy(roadmap)
    producer = changed["tasks"][1]
    producer["prompt"] = producer["prompt"].replace("sentence_protocol_ready_score", "wrong", 1)
    assert not subject.compare_contract(design, changed)["passed"]


def test_prior_custody_separates_queue_receipt() -> None:
    """SCENARIO-REPORT-7767-CUSTODY: a skip receipt is not producer science."""
    rows = subject.prior_inventory(subject.ROOT)
    assert len(rows) == 14
    absent = [row for row in rows if row["producer_state"] == "missing"]
    assert len(absent) == 4
    assert {7757, 7764} <= {row["experiment"] for row in absent if row["pre_gate_path"]}
    assert all(row["producer_sha256"] is None for row in absent)
    assert sum(row["producer_state"] == "disqualified" for row in rows) == 7
    assert sum(row["producer_state"] == "blocked" for row in rows) == 1


def test_independent_rows_and_custody(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7767-TERMINAL: raw rows and source bytes must replay."""
    root, design, roadmap = authorities
    (root / subject.DESIGN).parent.mkdir(parents=True)
    (root / subject.DESIGN).write_text(design)
    rows = subject.compare_contract(design, roadmap)["rows"]
    raw = root / "rows.json"
    raw.write_text(json.dumps(rows))
    sources = subject.source_hashes(root, Path("research-roadmap.yaml"), [subject.DESIGN])
    assert subject.cold_validate(root, raw, sources)
    altered = deepcopy(rows)
    altered[0]["matched"] = False
    raw.write_text(json.dumps(altered))
    assert not subject.cold_validate(root, raw, sources)
    raw.write_text(json.dumps(rows))
    (root / subject.DESIGN).write_text("changed")
    assert not subject.cold_validate(root, raw, sources)


def test_nested_basetemp_real_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7767-TERMINAL: nested pytest setup works in a child."""
    test_file = tmp_path / "test_child.py"
    test_file.write_text("def test_child():\n    assert True\n")
    basetemp = tmp_path / "nested" / "parent" / "child"
    basetemp.parent.mkdir(parents=True)
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
            str(test_file),
            "-q",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reject_missing_design_blocks_and_wrong_milestone(
    authorities: tuple[Path, str, dict],
) -> None:
    """SCENARIO-REPORT-7767-CONTRACT: malformed independent sources fail closed."""
    _, design, roadmap = authorities
    with pytest.raises(ValueError, match="JSON contract missing"):
        subject.parse_design(design.split("<!-- V676_TASK_CONTRACT_BEGIN -->")[0])
    wrong = design.replace('"milestone": "2026.09.676"', '"milestone": "2026.09.675"', 1)
    with pytest.raises(ValueError, match="JSON milestone"):
        subject.parse_design(wrong)
    changed = deepcopy(roadmap)
    changed["milestone"] = "2026.09.675"
    assert "roadmap_milestone" in subject.compare_contract(design, changed)["errors"]
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate(roadmap, "unknown")


def test_preserved_history_must_have_contract_and_order(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7767-CUSTODY: V675 history cannot be silently replaced."""
    path = tmp_path / subject.PRESERVED
    path.parent.mkdir(parents=True)
    path.write_text("no machine block")
    with pytest.raises(ValueError, match="machine contract missing"):
        subject.prior_inventory(tmp_path)
    preserved = (subject.ROOT / subject.PRESERVED).read_text()
    path.write_text(
        preserved.replace('"id":"exp7753-contract-methods"', '"id":"exp9999-contract-methods"', 1)
    )
    with pytest.raises(ValueError, match="task order changed"):
        subject.prior_inventory(tmp_path)


def test_failed_checks_report_every_operand(authorities: tuple[Path, str, dict]) -> None:
    """SCENARIO-REPORT-7767-TERMINAL: each failed input names observed evidence."""
    _, design, roadmap = authorities
    comparison = subject.compare_contract(design, subject.mutate(roadmap, "title"))
    sources = [
        {"path": str(subject.DESIGN), "role": "current_input", "exists": False, "sha256": None}
    ]
    failures = subject.failed_checks(comparison, sources)
    assert any(row["field"] == "exists" and row["observed"] is False for row in failures)
    assert any(
        row["field"] == "title" and row["upstream_id"] == roadmap["tasks"][0]["id"]
        for row in failures
    )
    assert subject.failed_checks(subject.compare_contract(design, roadmap), []) == []
