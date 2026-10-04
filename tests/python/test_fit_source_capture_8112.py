"""REQ-VERIFY-8112; REQ-REPORT-8112; SCENARIO-VERIFY-8112-PRIVATE.

Private transport fixtures test custody without performing model work.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import runpy

import pytest

from carnot import experiment_8112_v702_fit_source_capture as e
from carnot.verify import qwen_fit_source_capture_8112 as c
from carnot.reporting.current_work_receipt import atomic_json

_old = runpy.run_path(str(e.ROOT / "tests/python/test_fit_source_capture_8099.py"))
old_views, Runtime = _old["views"], _old["Runtime"]


def views():
    """Keep all source bytes, including sentence separators, observable."""
    value = old_views()
    for role in value.values():
        for row in role["request_rows"]:
            row["source_bytes"] = b"First sentence. Second sentence. Third sentence.".hex()
    return value


def test_frozen_interventions():
    """SCENARIO-VERIFY-8112-PRIVATE fixes the panel before any outcomes."""
    slots = c.freeze(views())
    assert len(slots) == 288 and slots == c.freeze(views())
    by = {(r["unit_id"], r["arm"]): r for r in slots}
    assert len({r["unit_id"] for r in slots[192:]}) == 24
    for i in range(24):
        full = by[(f"fit-{i}", "full_source")]
        assert full["request"] == by[(f"fit-{i}", "duplicate")]["request"]
        assert (
            by[(f"fit-{i}", "source_mismatched")]["intervention_source_id"] == f"fit-{(i + 1) % 24}"
        )
        perm = by[(f"fit-{i}", "source_order_permuted")]
        assert perm["source_permutation"] == [2, 1, 0]
        body = json.loads(perm["request"]["messages"][1]["content"])
        assert body["complete_source"] == "Third sentence. Second sentence. First sentence."
        assert body["original_answer"] == "Answer unchanged."
        assert c.config()["call_limit"] == 288
    with pytest.raises(ValueError):
        c.freeze({})


def test_capture_reparse_cache_and_pairing(tmp_path):
    """REQ-VERIFY-8112 keeps cached calls historical and pairs duplicate noise."""
    slots = c.freeze(views())
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "model/template/key", ledger=ledger)
    reduced = c.reduce(rows)
    assert reduced["completed_count"] == 288 and reduced["independent_count"] == 192
    assert reduced["source_effect_diagnostics"]["source_order_permuted"]["paired_count"] == 24
    assert reduced["duplicate_variation"]["parse_denominator"] == 24
    assert all(r["human_target"] is None for r in rows)
    fresh = c.prior.Ledger(tmp_path / "fresh.json")
    cached = c.capture(slots, Runtime(), tmp_path / "slots", "model/template/key", ledger=fresh)
    assert fresh.counts()["generation_calls_attempted"] == 0
    assert all(r["measurement_scope"] == "historical" for r in cached)
    with pytest.raises(ValueError):
        c.capture(slots, Runtime(), tmp_path / "slots", "changed-template", ledger=fresh)
    drift = deepcopy(rows)
    drift[0]["probability"] = 1
    with pytest.raises(ValueError):
        c.reduce(drift)
    with pytest.raises(ValueError):
        c.reduce(rows + rows[:1])
    partial = [r for r in rows if not (r["unit_id"] == "fit-0" and r["arm"] == "duplicate")]
    assert c.reduce(partial)["source_effect_diagnostics"]["source_removed"]["paired_count"] == 23


def test_class_support_original_only(tmp_path):
    """REQ-VERIFY-8112 counts only original scores with usable human labels."""
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(c.freeze(views()), Runtime(), tmp_path / "slots", "private", ledger=ledger)
    labels = {
        role: [dict(unit_id=f"{role}-{i}", y=i % 2) for i in range(n)]
        for role, n in c.ROLES.items()
    }
    assert c.support(rows, labels)["ready"] == 1
    labels["fit"] = [dict(r, y=0) for r in labels["fit"]]
    assert c.support(rows, labels)["ready"] == 0
    assert c.support([], {})["ready"] == 0
    blocked = c.capture(
        c.freeze(views())[:1],
        Runtime(),
        tmp_path / "blocked",
        "private",
        ledger=c.prior.Ledger(tmp_path / "b.json"),
        blocked_reason="missing_external",
    )
    assert not blocked[0]["started"] and blocked[0]["exclusion_reason"] == "missing_external"


@pytest.mark.parametrize("branch", ["zero", "load", "generation", "resumed", "owned", "fixture"])
def test_real_terminal_builder_provenance(tmp_path, branch):
    """SCENARIO-REPORT-8112-TERMINAL exercises the actual publication builder."""
    plan = dict(checks=[], references=[], slots=c.freeze(views()), protocol={}, manifests={})
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    result = dict(rows=[], ledger=[], checks=[], measured_duration_s=12)
    if branch == "zero":
        plan["checks"] = [e.operand("external", tmp_path / "absent", "exists", True, False)]
    if branch in ("load", "generation", "owned"):
        ledger.start("model_load", "load", {})
        ledger.finish("load", "failed" if branch == "load" else "completed", {})
    if branch in ("generation", "resumed", "fixture", "owned"):
        result["rows"] = c.capture(
            plan["slots"], Runtime(), tmp_path / "slots", "private", ledger=ledger
        )
        if branch == "resumed":
            ledger = c.prior.Ledger(tmp_path / "fresh.json")
            result["rows"] = c.capture(
                plan["slots"], Runtime(), tmp_path / "slots", "private", ledger=ledger
            )
        plan["labels"] = {
            role: [dict(unit_id=f"{role}-{i}", y=i % 2) for i in range(n)]
            for role, n in c.ROLES.items()
        }
    result.update(
        ledger=ledger.rows,
        model_identity_receipt=dict(
            authenticated=True, offload_layers=[1], props=dict(chat_template="private")
        ),
        gpu_lease_receipt=dict(owner_pid=os.getpid()),
    )
    value = e.build(
        plan,
        result,
        dict(passed=branch != "owned", receipts=[]),
        tmp_path,
        12,
        fixture=branch == "fixture",
    )
    expected = (
        "model_bounded_generation"
        if branch in ("generation", "owned")
        else "model_load_no_generation"
        if branch == "load"
        else "no_model_load"
    )
    assert value["inference_substrate_class"] == expected
    assert value["planned_MODEL_SPECS"] == [c.risk.MODEL]
    assert bool(value["MODEL_SPECS"]) == (expected != "no_model_load")
    assert value["fit_capture_ready_score"] == int(branch == "generation")
    assert value["verifier_is_oracle"] == (branch == "fixture")
    assert e.replay_value(value)
    drift = deepcopy(value)
    drift["inference_substrate_class"] = "model_full_generation"
    with pytest.raises(ValueError):
        e.replay_value(drift)


def test_preconditions_commands_and_live_reuse(tmp_path, monkeypatch):
    """REQ-REPORT-8112 checks exact operands and reuses the qualified worker."""
    plan = e.preconditions(tmp_path)
    assert not plan["slots"] and any(not r["passed"] for r in plan["checks"])
    commands = e.commands(tmp_path)
    assert commands[0]["name"] == "collect_consumers"
    assert "--collect-only" in commands[0]["argv"]
    assert all("test_primary_publication.py" not in a for r in commands for a in r["argv"])
    assert any("test_primary_publication_7928.py" in a for a in commands[0]["argv"])
    monkeypatch.setattr(e, "_live", lambda *a: dict(ledger=[], checks=[]))
    assert e.live_capture(plan, tmp_path, tmp_path)["owned_failure"] is False
    monkeypatch.setattr(
        e,
        "_live",
        lambda *a: dict(ledger=[dict(operation="model_load")], checks=[dict(passed=False)]),
    )
    assert e.live_capture(plan, tmp_path, tmp_path)["owned_failure"] is True


def test_private_script_routes(tmp_path):
    """SCENARIO-REPORT-8112-TERMINAL runs real CLIs outside the checkout."""
    fixture = tmp_path / "public.json"
    atomic_json(fixture, views())
    cli = e.ROOT / e.OWNED[2]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)

    def run(*args):
        return subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), str(cli), "--date", "20261004", *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    output = tmp_path / "success" / (e.NAME + ".json")
    assert run("--fixture", fixture, "--output", output).returncode == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive" and value["verifier_is_oracle"]
    assert run("--cold-replay", output).returncode == 0
    value["completed_count"] += 1
    atomic_json(output, value)
    assert run("--cold-replay", output).returncode == 1
    for name, exists in [("blocked", False), ("mutation", True)]:
        bad = tmp_path / name / "fixture.json"
        if exists:
            v = views()
            v["fit"]["request_rows"][0]["labels"] = []
            atomic_json(bad, v)
        target = bad.parent / (e.NAME + ".json")
        assert run("--fixture", bad, "--output", target).returncode == 0
        assert json.loads(target.read_text())["verdict_class"] == (
            "disqualified" if exists else "blocked"
        )


def test_main_validation_and_capture_routes(tmp_path, monkeypatch):
    """REQ-REPORT-8112 validates before model admission through inherited main."""
    monkeypatch.setattr(
        e,
        "preconditions",
        lambda root: dict(checks=[], references=[], slots=[], manifests={}, protocol={}),
    )
    monkeypatch.setattr(
        e.qualified,
        "validate",
        lambda raw, private: dict(passed=True, receipts=[], global_health=[]),
    )
    monkeypatch.setattr(
        e,
        "_run_check",
        lambda *a, **k: dict(
            passed=True,
            actual_exit=0,
            timed_out=False,
            output_tail="CUDA",
            log_path=str(__file__),
            log_sha256=e.sha256_file(Path(__file__)),
        ),
    )
    monkeypatch.setattr(e, "live_capture", lambda *a: dict(rows=[], checks=[], ledger=[]))
    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--date", "20261004", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["inference_substrate_class"] == "no_model_load"


def test_authenticated_preconditions_and_evaluator_drift(tmp_path):
    """REQ-REPORT-8112 authenticates existing terminals and original targets."""
    plan = e.preconditions(e.ROOT)
    assert len(plan["slots"]) == 288 and all(r["passed"] for r in plan["checks"])
    assert set(plan["evaluator_refs"]) == set(c.ROLES)
    label_path = tmp_path / "labels.json"
    atomic_json(label_path, dict(rows=[]))
    private = dict(
        plan, evaluator_refs=dict(fit=dict(path=str(label_path), sha256=e.sha256_file(label_path)))
    )
    value = e.build(
        private,
        dict(rows=[], ledger=[], checks=[]),
        dict(passed=True, receipts=[]),
        tmp_path,
        1,
        fixture=False,
    )
    assert value["fit_capture_ready_score"] == 0
    atomic_json(label_path, dict(rows=[dict(unit_id="changed", y=1)]))
    with pytest.raises(ValueError, match="evaluator_hash_drift"):
        e.build(
            private,
            dict(rows=[], ledger=[], checks=[]),
            dict(passed=True, receipts=[]),
            tmp_path,
            1,
            fixture=False,
        )


def test_altered_truth_is_rejected(tmp_path):
    """REQ-VERIFY-8112 never transfers labels to altered evidence."""
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(
        c.freeze(views())[192:193], Runtime(), tmp_path / "slots", "private", ledger=ledger
    )
    rows[0]["human_target"] = 0
    with pytest.raises(ValueError, match="altered_truth_label"):
        c.reduce(rows)


def test_runtime_template_keys_and_real_progress(tmp_path, monkeypatch, capsys):
    """REQ-REPORT-8112 binds resumed keys to authenticated runtime bytes."""
    atomic_json(
        tmp_path / "ledger.json", dict(rows=[dict(operation="generation", status="completed")])
    )
    template = "private native template"
    plan = dict(
        protocol=dict(
            runtime_sha256="runtime", chat_template_sha256=e.c.prior.canonical_hash(template)
        )
    )

    def loaded(*args):
        e.qualified.legacy.progress("owned_runtime_heartbeat", 0)
        return dict(
            ledger=[dict(operation="model_load")],
            checks=[],
            native_binary=dict(sha256="runtime"),
            model_identity_receipt=dict(authenticated=True, props=dict(chat_template=template)),
        )

    monkeypatch.setattr(e, "_live", loaded)
    assert not e.live_capture(plan, tmp_path, tmp_path)["owned_failure"]
    assert "completed_units=1 pending_units=287" in capsys.readouterr().out
    plan["protocol"]["runtime_sha256"] = "changed"
    assert e.live_capture(plan, tmp_path, tmp_path)["owned_failure"]
    (tmp_path / "ledger.json").unlink()
    assert e.live_capture(plan, tmp_path, tmp_path)["owned_failure"]


def test_qualified_revision_comes_from_runtime_receipt():
    """REQ-REPORT-8112 reads actual Exp8102 provenance rather than an absent alias."""
    historical = json.loads((e.ROOT / e.HISTORY).read_text())
    plan = e.preconditions(e.ROOT)
    assert plan["protocol"]["model_revision"] == historical["runtime_identity"]["model_revision"]
    assert plan["protocol"]["model_revision"] is not None
