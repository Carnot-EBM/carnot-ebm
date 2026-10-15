"""REQ-REPORT-8526: complete authority and unchanged scientific operands."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot.reporting import v733_contract_methods as e


@pytest.fixture
def full_root(tmp_path):
    """Use the sealed actual activation without fabricating staging evidence."""
    root = tmp_path / "authority"
    p = e.protocol()
    for name, key in [(e.DESIGN, "v733_design_reference"), (e.ACTIVE, "v733_activation_reference")]:
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
    """SCENARIO-REPORT-8526-AUTHORITY: twelve independent mutations must fail."""
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
        plan["tasks"][1]["gated_on"][0]["value"] = 0
    if mode == "retirement":
        task["prior_failures"][0]["retire_if_same_verdict"] = False
    if mode == "count":
        plan["tasks"].pop()
    if mode == "order":
        plan["tasks"][:2] = plan["tasks"][1::-1]
    if mode == "end":
        design.write_text(design.read_text().replace("<!-- V733_TASK_CONTRACT_END -->", ""))
    if mode == "missing":
        design.unlink()
    if mode == "rehash":
        task["prompt"] += " forged"
        design.write_text(
            design.read_text()
            .split("<!-- V733_TASK_CONTRACT_START")[0]
            .replace(e.TASKS_PIN, e.tasks_digest(plan["tasks"]))
            + "<!-- V733_TASK_CONTRACT_START -->\n```json\n"
            + json.dumps(plan)
            + "\n```\n<!-- V733_TASK_CONTRACT_END -->\n"
        )
    if mode == "stage":
        staged = deepcopy(plan)
        staged["tasks"][0]["prompt"] += " changed"
        (full_root / e.STAGED).write_text(yaml.safe_dump(staged))
    active.write_text(yaml.safe_dump(plan))
    result = e.authority(full_root, tmp_path / "snapshots")
    assert result["activated"] == (mode == "valid")
    assert len(result["contract_rows"]) == 14
    assert all(row["effective_independent_groups"] == 0 for row in result["contract_rows"])
    if mode != "valid":
        assert result["gate_check_summary"]
        assert all(g["upstream_id"] == e.TASK for g in result["gate_check_summary"])


def test_protocol():
    """SCENARIO-REPORT-8526-METHODS: old science and new workloads stay distinct."""
    p = e.protocol()
    old = e.previous.protocol()
    assert all(p[k] == old[k] for k in e.INHERITED)
    assert e.tasks_digest(p["task_map_v733"]) == e.TASKS_PIN
    assert [len(v) for v in p["partition"]["roles"].values()] == [150, 100, 50]
    assert p["identity_brier"]["full_coefficients"] == 109
    assert p["identity_brier"]["scalar_coefficients"] == 13
    native, arc = p["nine_input_native_inference"], p["arc_capability_mask"]
    assert native["inputs"] == 9 and native["coefficients"] == 109
    assert native["paired_repeats"] == 10 and native["bootstrap_seed"] == 7338534
    assert native["p95_ratio_at_most"] == 1.05 and native["nfr01_lower95_at_least"] == 10
    assert not native["historical_five_input_service_is_prerequisite"]
    assert arc["episodes"] == 16 and arc["model_calls"] == 0
    assert arc["refinement_min_resolved_per_shared_cell"] == 5
    assert p["retirement_ledger"]["V717"]["H1"] == -0.00390625
    assert len(p["v732_scheduling_dispositions"]) == 14


@pytest.mark.parametrize(
    "mode", ["note", "inherit", "native", "arc", "tasks", "preserved", "activation"]
)
def test_protocol_rejection(monkeypatch, mode):
    """SCENARIO-REPORT-8526-METHODS: self-rehashed method envelopes cannot move seals."""
    original_read, original_hash = e.read, e.sha256_file
    if mode in {"note", "preserved", "activation"}:
        key = {
            "note": e.METHODS,
            "preserved": "research-roadmap-v732-preserved-20261014.md",
            "activation": "unused",
        }[mode]
        if mode == "activation":
            key = Path(e.protocol()["v733_activation_reference"]["path"]).name
        monkeypatch.setattr(
            e, "sha256_file", lambda p: "changed" if p.name == Path(key).name else original_hash(p)
        )
    else:

        def changed(ref):
            value = deepcopy(original_read(ref))
            if ref["path"] == str(e.ROOT / e.PROTOCOL):
                if mode == "inherit":
                    value["identity_brier"]["steps"] = 199
                elif mode == "native":
                    value["nine_input_native_inference"]["inputs"] = 5
                elif mode == "arc":
                    value["arc_capability_mask"]["episodes"] = 8
                else:
                    value["task_map_v733"][0]["prompt"] += " forged"
            return value

        monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError):
        e.protocol()
