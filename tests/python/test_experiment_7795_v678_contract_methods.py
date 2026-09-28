"""REQ-REPORT-7795: V678 authority and custody use frozen real bytes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7795_v678_contract_methods as subject


@pytest.fixture
def authorities() -> tuple[str, dict]:
    """Use authentic 2026-09-27 snapshots, independent of the active roadmap."""
    root = subject.ROOT
    return (
        (root / subject.DESIGN_SNAPSHOT).read_text(),
        yaml.safe_load((root / subject.YAML_SNAPSHOT).read_text()),
    )


def test_three_authorities_and_sequence(authorities: tuple[str, dict]) -> None:
    """SCENARIO-REPORT-7795-CONTRACT: each task and field agrees."""
    design, roadmap = authorities
    result = subject.compare_contract(design, roadmap)
    assert result["passed"]
    assert len(result["rows"]) == 14
    assert [row["unit_id"].split("-")[0] for row in result["rows"]] == [
        f"exp{number}" for number in range(7795, 7809)
    ]
    assert all(all(row["checks"].values()) for row in result["rows"])


@pytest.mark.parametrize(
    "mutation",
    [
        "drop",
        "reorder",
        "title",
        "phase",
        "deliverable",
        "model",
        "substrate",
        "unknown_producer",
        "gate_field",
        "prior_experiment_id",
        "prior_verdict",
        "prior_addressed_by",
        "prior_retirement",
    ],
)
def test_private_yaml_mutations_fail(authorities: tuple[str, dict], mutation: str) -> None:
    """SCENARIO-REPORT-7795-CONTRACT: private changes cannot pass."""
    design, roadmap = authorities
    changed = subject.mutate(roadmap, mutation)
    assert not subject.compare_contract(design, changed)["passed"]
    assert roadmap != changed


def test_private_design_mutations_fail(authorities: tuple[str, dict]) -> None:
    """SCENARIO-REPORT-7795-CONTRACT: stale design is independently read."""
    design, roadmap = authorities
    stale = design.replace(
        "Bind fourteen tasks and register source dependence methods", "Stale contract title", 1
    )
    assert not subject.compare_contract(stale, roadmap)["passed"]
    changed = design.replace(
        '"MODEL_SPECS": ["unsloth/Qwen3.8-27B-GGUF"]', '"MODEL_SPECS": ["wrong/model"]', 1
    )
    assert not subject.compare_contract(changed, roadmap)["passed"]


def test_resolution_and_cold_snapshot(tmp_path: Path, authorities: tuple[str, dict]) -> None:
    """SCENARIO-REPORT-7795-CONTRACT: rollover cannot change frozen authority."""
    design, roadmap = authorities
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(roadmap))
    selected, actual, candidates = subject.resolve_authority(tmp_path)
    assert selected.name == "research-roadmap.yaml" and actual == roadmap
    assert candidates[0]["exists"] is False
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(roadmap))
    assert subject.resolve_authority(tmp_path)[0].name == "research-roadmap-next.yaml"
    changed = deepcopy(roadmap)
    changed["milestone"] = "2026.09.679"
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(changed))
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="V678 authority"):
        subject.resolve_authority(tmp_path)
    assert subject.compare_contract(design, roadmap)["passed"]


def test_v677_declared_producer_custody() -> None:
    """SCENARIO-REPORT-7795-CUSTODY: queue receipts are not producers."""
    rows = subject.prior_inventory(subject.ROOT)
    assert len(rows) == 14
    assert sum(row["producer_state"] == "missing" for row in rows) == 6
    assert sum(row["producer_state"] != "missing" for row in rows) == 8
    assert [row["experiment"] for row in rows if row["producer_state"] == "missing"] == [
        7783,
        7785,
        7786,
        7788,
        7791,
        7792,
    ]
    for row in rows:
        if row["pre_gate_path"]:
            assert row["producer_state"] == "missing"
            assert row["pre_gate_path"] != row["producer_path"]


def test_terminal_candidate_fails_changed_rows(
    tmp_path: Path, authorities: tuple[str, dict]
) -> None:
    """SCENARIO-REPORT-7795-TERMINAL: a fresh reader detects changed rows."""
    design, roadmap = authorities
    rows = subject.compare_contract(design, roadmap)["rows"]
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(rows))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"rows": rows}))
    assert subject.cold_validate(candidate, raw, subject.ROOT)
    rows[0]["matched"] = False
    candidate.write_text(json.dumps({"rows": rows}))
    assert not subject.cold_validate(candidate, raw, subject.ROOT)


def test_malformed_embedded_contract_and_milestone(authorities: tuple[str, dict]) -> None:
    """SCENARIO-REPORT-7795-CONTRACT: missing JSON and wrong milestone refuse."""
    design, roadmap = authorities
    with pytest.raises(ValueError, match="JSON contract missing"):
        subject.parse_design(design.replace("V678_TASK_CONTRACT_START", "BROKEN_START"))
    with pytest.raises(ValueError, match="JSON milestone mismatch"):
        subject.parse_design(
            design.replace('"milestone": "2026.09.678"', '"milestone": "2026.09.679"')
        )
    changed = deepcopy(roadmap)
    changed["milestone"] = "2026.09.679"
    assert "roadmap_milestone" in subject.compare_contract(design, changed)["errors"]
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate(roadmap, "unknown")


def test_v677_order_and_missing_input_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7795-CUSTODY: changed history and absent bytes refuse."""
    source = subject.ROOT / "docs/research-notes/v677-authority-snapshots/roadmap.yaml"
    archived = yaml.safe_load(source.read_text())
    archived["tasks"][0], archived["tasks"][1] = archived["tasks"][1], archived["tasks"][0]
    path = tmp_path / "docs/research-notes/v677-authority-snapshots/roadmap.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(archived))
    with pytest.raises(ValueError, match="V677 task order"):
        subject.prior_inventory(tmp_path)
    present = tmp_path / "present.txt"
    present.write_text("real bytes")
    sources = subject.source_hashes(tmp_path, [Path("present.txt"), Path("missing.txt")])
    assert sources[0]["sha256"] and sources[0]["eligible"]
    assert sources[1]["sha256"] is None and not sources[1]["eligible"]


def test_cold_reader_needs_both_snapshots(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7795-TERMINAL: an absent snapshot cannot replay."""
    candidate = tmp_path / "candidate.json"
    raw = tmp_path / "rows.json"
    candidate.write_text('{"rows": []}')
    raw.write_text("[]")
    assert not subject.cold_validate(candidate, raw, tmp_path)
