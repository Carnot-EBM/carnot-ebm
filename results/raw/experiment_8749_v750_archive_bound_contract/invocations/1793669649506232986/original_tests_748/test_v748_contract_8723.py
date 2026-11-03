"""REQ-REPORT-8723: V748 grants administrative eligibility only."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import yaml

from carnot.reporting import v748_collection_contract as api
from carnot.reporting.v709_execution import child


def test_historical_bindings_remain_isolated(tmp_path, capsys):
    """REQ-REPORT-8723: current ownership must not change historical helper identities."""
    from carnot.reporting import v747_contract_methods as historical

    assert api.m is not historical
    path = tmp_path / "absent"
    original_gate = historical.gate(path, "authority", True, False)
    current_gate = api.m.gate(path, "authority", True, False)
    assert original_gate["upstream"] == "exp8709-contract-methods"
    assert current_gate["upstream"] == "exp8723-collection-contract"
    historical.progress("isolated", 1, 0)
    api.m.progress("isolated", 1, 0)
    assert capsys.readouterr().out.splitlines() == [
        "[exp8709] phase=isolated completed=1 pending=0",
        "[exp8723] phase=isolated completed=1 pending=0",
    ]


def test_branch_authority_and_original_methods(tmp_path):
    """SCENARIO-REPORT-8723-AUTHORITY: every branch binds its own unchanged operands."""
    m = api.m
    authority = m.authority(m.ROOT)
    assert authority["ready"] and len(authority["rows"]) == 14
    assert [r["task_object"]["id"] for r in authority["rows"]][0] == m.TASK
    assert not m.authority(tmp_path)["ready"]
    rows = m.mutations()
    assert len(rows) >= 20 and all(r["passed"] for r in rows)
    retained, methods = m.retained(), m.methods()
    assert retained["passed"] and retained["qualification"]["original_node_count"] == 77
    assert methods["learning_authority_inputs_ready_score"] == 1
    original = m.original.protocol()
    for key in ["objective_updates", "objective_audit", "installed_batching_protocol"]:
        assert methods[key] == original[key]
    assert len(methods["branch_method_bindings"]) == 14
    assert [len(c["node_ids"]) for c in methods["original_partition"]["cohorts"]] == [39, 25]
    assert methods["learning_core_execution_owner"] == "exp8724-invocation-clock-core"


@pytest.mark.parametrize(
    "field", ["agent_type", "routing", "prompt", "gated_on", "deliverable", "MODEL_SPECS"]
)
def test_full_contract_dimensions(tmp_path, field):
    """REQ-REPORT-8723: even fields outside the visible table require exact equality."""
    m = api.m
    text = (m.ROOT / m.DESIGN).read_text()
    tasks = m.parse_design(text, milestone=m.MILESTONE)[1]
    tasks[0][field] = "forged"
    design = tmp_path / m.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(text)
    (tmp_path / m.ACTIVE).write_text(yaml.safe_dump(dict(milestone=m.MILESTONE, tasks=tasks)))
    assert not m.authority(tmp_path)["ready"]


@pytest.mark.parametrize("source", ["active", "staged", "both"])
def test_conductor_prompt_encoding(tmp_path, source):
    """SCENARIO-REPORT-8723-PROMPT-ENCODING: bind encoding without changing authority."""
    m = api.m
    text = (m.ROOT / m.DESIGN).read_text()
    tasks = m.parse_design(text, milestone=m.MILESTONE)[1]
    encoded = deepcopy(tasks)
    prompt = encoded[7]["prompt"]
    encoded[7]["prompt"] = prompt.replace('{"repo":', '{{"repo":').replace(
        '"mandated_local_headline_model"}', '"mandated_local_headline_model"}}'
    )
    assert encoded != tasks
    design = tmp_path / m.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(text)
    active = tmp_path / m.ACTIVE
    active.write_text(
        yaml.safe_dump(dict(milestone=m.MILESTONE, tasks=tasks if source == "staged" else encoded))
    )
    if source != "active":
        (tmp_path / m.STAGED).write_text(yaml.safe_dump(dict(milestone=m.MILESTONE, tasks=encoded)))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    for _ in range(2):
        result = m.authority(tmp_path)
        assert result["ready"]
        assert [row["task_object"] for row in result["rows"]] == tasks
        assert all(
            m.e.sha256_file(Path(ref["path"])) == ref["sha256"] for ref in result["references"]
        )
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize("mode", ["date", "project_root", "extra_braces", "content", "field"])
def test_prompt_encoding_rejects_changes(tmp_path, mode):
    """REQ-REPORT-8723: template encoding cannot conceal changes to the sealed task."""
    m = api.m
    text = (m.ROOT / m.DESIGN).read_text()
    tasks = m.parse_design(text, milestone=m.MILESTONE)[1]
    task = tasks[7]
    task["prompt"] = (
        task["prompt"]
        .replace('{"repo":', '{{"repo":')
        .replace('"mandated_local_headline_model"}', '"mandated_local_headline_model"}}')
    )
    if mode in {"date", "project_root"}:
        task["prompt"] = task["prompt"].replace("{" + mode + "}", "{{" + mode + "}}")
    elif mode == "extra_braces":
        task["prompt"] = (
            task["prompt"]
            .replace('{{"repo":', '{{{{"repo":')
            .replace('"mandated_local_headline_model"}}', '"mandated_local_headline_model"}}}}')
        )
    elif mode == "content":
        task["prompt"] = task["prompt"].replace("Q4_K_M", "Q8_0")
    else:
        task["routing"] = "forged"
    design = tmp_path / m.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(text)
    (tmp_path / m.ACTIVE).write_text(yaml.safe_dump(dict(milestone=m.MILESTONE, tasks=tasks)))
    assert not m.authority(tmp_path)["ready"]


def test_private_cli_and_sealed_replay(tmp_path):
    """SCENARIO-REPORT-8723-AUTHORITY: real children keep private readiness at zero."""
    m = api.m
    output = tmp_path / "results" / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("date", ["--date", "20261101"], 2),
        ("unsafe", ["--private-control"], 2),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipt = child(name, command + args, tmp_path / "logs", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8723 and value["verdict_class"] == "blocked"
    assert value["learning_authority_ready_score"] == value["contract_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and api.replay(output)
    original = deepcopy(value)
    value["learning_authority_ready_score"] = 1
    m.e.atomic_json(output, value)
    assert not api.replay(output)
    m.e.atomic_json(output, original)
    work = m.e.load(original["primitive_reference"])
    work["contract"]["ready"] = False
    path = Path(original["primitive_reference"]["path"])
    m.e.atomic_json(path, work)
    original["primitive_reference"] = m.e.reference(path)
    seal = Path(original["terminal_validation_sidecar_path"]).parent / "independent_seal.json"
    m.e.atomic_json(seal, {k: original[k] for k in ["primitive_reference", "validation_reference"]})
    m.e.atomic_json(output, original)
    assert not api.replay(output)
    path.unlink()
    assert not api.replay(output)


def test_missing_preserved_operand(monkeypatch):
    """REQ-REPORT-8723: missing historical design blocks rather than becoming zero data."""
    original = api.m.original.checked
    monkeypatch.setattr(
        api.m.original, "checked", lambda ref: False if ref == api.PRESERVED else original(ref)
    )
    methods = api.m.methods()
    assert methods["learning_authority_inputs_ready_score"] == 0
    assert methods["gate_check_summary"][-1]["field"] == "preserved_v747_design"


def test_absent_priors_are_bound_not_invented(tmp_path):
    """REQ-REPORT-8723: a new branch may have no recorded failure; its exact object binds."""
    m = api.m
    text = (m.ROOT / m.DESIGN).read_text()
    tasks = m.parse_design(text, milestone=m.MILESTONE)[1]
    assert "prior_failures" not in tasks[3] and "prior_failures" not in tasks[8]
    design = tmp_path / m.DESIGN
    design.parent.mkdir(parents=True)
    design.write_text(text)
    roadmap = tmp_path / m.ACTIVE
    roadmap.write_text(yaml.safe_dump(dict(milestone=m.MILESTONE, tasks=tasks)))
    assert m.authority(tmp_path)["ready"]
    tasks[0].pop("prior_failures")
    roadmap.write_text(yaml.safe_dump(dict(milestone=m.MILESTONE, tasks=tasks)))
    assert not m.authority(tmp_path)["ready"]


def test_original_assertions_and_failed_child(tmp_path, monkeypatch):
    """REQ-VERIFY-8723: execute the original full-path assertions under current bindings."""
    import runpy

    original = runpy.run_path(str(api.m.ROOT / "tests/python/test_v747_contract_8709.py"))
    check = original["test_collection_and_full_path"]
    monkeypatch.setitem(check.__globals__, "m", api.m)
    monkeypatch.setitem(check.__globals__, "runner", api.runtime)
    with monkeypatch.context() as nested:
        check(tmp_path, nested)
    output = tmp_path / "failed" / (api.m.NAME + ".json")
    assert api.main(["--failed-child", "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["flagged_adversarial"]
    assert api.replay(output)


@pytest.mark.parametrize(
    "name,mode",
    [
        ("test_retained_rejection", mode)
        for mode in ["missing", "changed", "field", "zero", "flagged", "terminal"]
    ]
    + [("test_retained_deep_identity", mode) for mode in ["source", "qualification", "seal"]]
    + [
        (name, None)
        for name in [
            "test_direct_replay_controls",
            "test_missing_original_methods",
            "test_replay_rejects_historical_child_substitution",
        ]
    ],
)
def test_original_rejection_controls(tmp_path, monkeypatch, name, mode):
    """REQ-VERIFY-8723: preserve unchanged historical rejection assertions under V748 bindings."""
    import inspect
    import runpy

    original = runpy.run_path(str(api.m.ROOT / "tests/python/test_v747_contract_8709.py"))
    check = original[name]
    monkeypatch.setitem(check.__globals__, "m", api.m)
    monkeypatch.setitem(check.__globals__, "runner", api.runtime)
    arguments = dict(tmp_path=tmp_path, monkeypatch=monkeypatch, mode=mode)
    check(**{key: arguments[key] for key in inspect.signature(check).parameters})
