"""REQ-REPORT-9166-V781: activated queue binding preserves exact design semantics."""

from copy import deepcopy
import json

import pytest
import yaml

from carnot.reporting import v781_contract_evidence as e


def test_activated_queue_pin_and_prompt_rendering():
    """SCENARIO-REPORT-9166-V781: bind the queue after literal braces are encoded."""
    text = (e.ROOT / e.DESIGN).read_text()
    machine = json.loads(
        text.split("## Full machine-readable contract", 1)[1]
        .split("```json", 1)[1]
        .split("```", 1)[0]
    )
    queue = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())
    assert e.ref(e.ROOT / e.ACTIVE)["sha256"] == e.FROZEN["pins"][e.ACTIVE]
    assert queue["tasks"] == [dict(t, prompt=e.encode(t["prompt"])) for t in machine]
    for task, original in zip(queue["tasks"], machine, strict=True):
        assert task["prompt"].format(project_root="/workspace", date="20261117") == (
            original["prompt"].replace("{project_root}", "/workspace").replace("{date}", "20261117")
        )


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (0, "prompt", "changed {project_root} {date}"),
        (0, "MODEL_SPECS", [{"hf_id": "changed"}]),
        (1, "gated_on", []),
    ],
)
def test_matching_queue_pin_cannot_bless_changed_task(tmp_path, monkeypatch, index, field, value):
    """SCENARIO-REPORT-9166-V781: exact design objects remain a separate gate."""
    design = tmp_path / e.DESIGN
    design.parent.mkdir(parents=True)
    design.write_bytes((e.ROOT / e.DESIGN).read_bytes())
    queue = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())
    queue["tasks"][index][field] = value
    active = tmp_path / e.ACTIVE
    active.write_text(yaml.safe_dump(queue))
    frozen = deepcopy(e.FROZEN)
    frozen["pins"][e.ACTIVE] = e.ref(active)["sha256"]
    monkeypatch.setattr(e, "FROZEN", frozen)
    result = e.authority(tmp_path)
    assert all(r["passed"] for r in result["checks"] if r["check"] == "contract_bytes")
    assert not result["ready"]
    assert not result["checks"][-1]["passed"]
