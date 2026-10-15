"""REQ-REPORT-8552: complete authority and unchanged scientific operands."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v735_contract_methods as e


@pytest.fixture
def full_root(tmp_path):
    """Use the sealed actual activation without fabricating staging evidence."""
    root = tmp_path / "authority"
    p = e.protocol()
    for name, key in [(e.DESIGN, "v735_design_reference"), (e.ACTIVE, "v735_activation_reference")]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(p[key]["path"]).read_bytes())
    return root


@pytest.mark.parametrize(
    "mode",
    [
        "valid",
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "model",
        "substrate",
        "gate",
        "prompt",
        "retirement",
        "end",
        "missing",
        "rehash",
        "stage",
    ],
)
def test_authority(full_root, tmp_path, mode):
    """SCENARIO-REPORT-8552-AUTHORITY: thirteen independent mutations must fail."""
    active, design = full_root / e.ACTIVE, full_root / e.DESIGN
    plan = yaml.safe_load(active.read_bytes())
    task = plan["tasks"][0]
    if mode in {"id", "title", "deliverable", "model", "prompt"}:
        task[mode] += " changed"
    if mode == "phase":
        task["phase"] = 4
    if mode == "substrate":
        task["inference_substrate_class"] = "model_bounded_generation"
    if mode == "gate":
        plan["tasks"][2]["gated_on"][0]["value"] = 0
    if mode == "retirement":
        task["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "count":
        plan["tasks"].pop()
    if mode == "order":
        plan["tasks"][:2] = plan["tasks"][1::-1]
    if mode == "end":
        design.write_text(design.read_text().replace("<!-- V735_TASK_CONTRACT_END -->", ""))
    if mode == "missing":
        design.unlink()
    if mode == "rehash":
        task["prompt"] += " forged"
        design.write_text(
            design.read_text()
            .split("<!-- V735_TASK_CONTRACT_START")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(plan["tasks"]))
            + "<!-- V735_TASK_CONTRACT_START -->\n```json\n"
            + json.dumps(plan)
            + "\n```\n<!-- V735_TASK_CONTRACT_END -->\n"
        )
    if mode == "stage":
        staged = deepcopy(plan)
        staged["tasks"][0]["prompt"] += " changed"
        (full_root / e.STAGED).write_text(yaml.safe_dump(staged))
    active.write_text(yaml.safe_dump(plan))
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 13
    assert all(row["effective_independent_groups"] == 0 for row in result["contract_rows"])
    if mode != "valid":
        assert result["gate_check_summary"]
        assert all(g["upstream_id"] == e.TASK for g in result["gate_check_summary"])


def test_protocol():
    """SCENARIO-REPORT-8552-METHODS: inherited numerical operands remain exact."""
    p = e.protocol()
    old = e.read(p["inherited_v734_protocol_reference"])
    assert all(p[k] == old[k] for k in e.INHERITED)
    assert len(p["task_map_v735"]) == 13
    assert p["preserved_v734_design_reference"]["sha256"] == e.DESIGN734_PIN
    assert p["closed_source_comparison"]["new_fits"] == 0
    assert [len(v) for v in p["partition"]["roles"].values()] == [150, 100, 50]


@pytest.mark.parametrize("mode", ["protocol", "inherit", "tasks", "preserved", "note"])
def test_protocol_rejection(monkeypatch, mode):
    """SCENARIO-REPORT-8552-METHODS: rehashing cannot replace independent seals."""
    original_read, original_hash = e.read, e.sha256_file
    if mode in {"preserved", "note"}:
        name = (
            e.METHODS if mode == "note" else e.protocol()["preserved_v734_design_reference"]["path"]
        )
        monkeypatch.setattr(
            e, "sha256_file", lambda p: "changed" if p.name == Path(name).name else original_hash(p)
        )
    else:

        def changed(ref):
            value = deepcopy(original_read(ref))
            if ref["path"] == str(e.ROOT / e.PROTOCOL):
                if mode == "protocol":
                    value["objective_updates"]["arms"] = []
                elif mode == "inherit":
                    value["partition"]["roles"] = {}
                else:
                    value["task_map_v735"][0]["prompt"] += "forged"
            return value

        monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()


def test_inherited_protocol_identity(monkeypatch):
    """SCENARIO-REPORT-8552-METHODS: a repaired envelope cannot substitute another producer."""
    p = e.protocol()
    old = e.read(p["inherited_v734_protocol_reference"])
    p["inherited_v734_protocol_reference"]["sha256"] = "forged"
    monkeypatch.setattr(
        e, "read", lambda ref: deepcopy(p) if ref["path"] == str(e.ROOT / e.PROTOCOL) else old
    )
    with pytest.raises(ValueError, match="original_protocol_pin"):
        e.protocol()
