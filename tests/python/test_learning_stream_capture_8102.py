"""REQ-VERIFY-8102; REQ-REPORT-8102; SCENARIO-REPORT-8102-CLI."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.verify import qwen_learning_stream_capture_8102 as c
from carnot import experiment_8102_v701_learning_stream_capture as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


class Runtime:
    """A scripted public transport fixture never represents live model work."""

    def count(self, text):
        return 30

    def generate(self, request):
        return dict(
            model=c.risk.MODEL,
            choices=[
                dict(
                    message=dict(
                        content=json.dumps(
                            dict(unsupported_probability=0.25, source_sentence_id=None)
                        )
                    ),
                    finish_reason="stop",
                )
            ],
            usage=dict(prompt_tokens=30, completion_tokens=18),
        )


def views():
    """Use private public sources; fixture transport earns no model credit."""
    return {
        role: dict(
            request_rows=[
                dict(
                    family_id=f"{role}-{i}",
                    source_bytes=f"Source {role} {i}.".encode().hex(),
                    answer_bytes=b"Answer unchanged.".hex(),
                )
                for i in range(n)
            ],
            roster=[
                dict(
                    family_id=f"{role}-{i}",
                    unit_id=f"{role}-{i}",
                    source_id=f"{role}-{i}",
                    source_cluster_id=f"cluster-{role}-{i}",
                    role=role,
                    order=i + (320 if role == "stream" else 576),
                    slot=i + 1,
                )
                for i in range(n)
            ],
        )
        for role, n in c.ROLES.items()
    }


def test_public_contract_and_future_label_rejection():
    slots = c.freeze(views())
    assert len(slots) == 320
    assert slots[0]["slot"] == 1 and slots[255]["slot"] == 256
    assert c.config()["output_tokens"] == 30720
    assert c.config()["latest_launch_s"] == 3150
    assert all(s["request"]["max_tokens"] == 96 for s in slots)
    for path, field in [("request_rows", "y"), ("roster", "future_label")]:
        bad = views()
        bad["stream"][path][0][field] = 1
        with pytest.raises(ValueError):
            c.freeze(bad)
    for bad in [{}, {"stream": views()["stream"]}]:
        with pytest.raises(ValueError):
            c.freeze(bad)
    bad = views()
    bad["retention"]["roster"][0]["source_cluster_id"] = "cluster-stream-0"
    with pytest.raises(ValueError):
        c.freeze(bad)
    bad = views()
    bad["stream"]["roster"].pop()
    with pytest.raises(ValueError):
        c.freeze(bad)


def test_capture_features_and_original_clock(tmp_path):
    slots = c.freeze(views())
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger)
    reduced = c.reduce(rows)
    assert reduced["stream_capture_ready_score"] == 1
    assert reduced["completed_count"] == 320
    assert reduced["independent_count"] == 320
    assert len(reduced["stream_features"]) == 256
    assert len(reduced["retention_features"]) == 64
    assert len(reduced["stream_features"][0]["values"]) == 9
    assert c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger) == rows
    missing = deepcopy(rows)
    missing[0].update(raw_response={}, status="failed")
    missing[0]["parsed"] = c.risk.transport.parse_response({}, missing[0]["visible_ids"])
    reduced = c.reduce(missing)
    assert reduced["stream_features"][0]["values"] is None
    assert reduced["stream_features"][1]["release_slot"] == 22
    assert reduced["original_slot_mask"]["stream"][:2] == [False, True]
    with pytest.raises(ValueError):
        c.reduce(rows + rows[:1])
    drift = deepcopy(rows)
    drift[0]["parsed"]["probability"] = 1
    with pytest.raises(ValueError):
        c.reduce(drift)
    drift = deepcopy(rows)
    drift[0]["lexical_feature"]["values"][0] = 99
    with pytest.raises(ValueError, match="lexical_feature_drift"):
        c.reduce(drift)


def test_no_retry_and_unstarted_budget(tmp_path):
    slots = c.freeze(views())[:2]
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    ledger.start("generation", slots[0]["family_id"], slots[0]["request"])
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger, deadline_s=0)
    assert rows[0]["status"] == "failed" and rows[1]["status"] == "censored"
    assert ledger.counts()["generation_calls_attempted"] == 1
    assert c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger) == rows
    with pytest.raises(ValueError):
        c.capture(slots, Runtime(), tmp_path / "slots", "changed", ledger=ledger)


def test_build_readiness_and_cold_mutation(tmp_path):
    slots = c.freeze(views())
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger)
    plan = dict(checks=[], references=[], slots=slots, capture_identity="private", manifests={})
    result = dict(rows=rows, ledger=ledger.rows, checks=[])
    value = e.build(plan, result, dict(passed=True, receipts=[]), tmp_path, 1, fixture=True)
    assert value["stream_capture_ready_score"] == 0
    assert value["verdict_class"] == "circular_positive"
    assert e.replay_value(value)
    bad = deepcopy(value)
    bad["completed_count"] -= 1
    with pytest.raises(ValueError):
        e.replay_value(bad)
    bad = deepcopy(value)
    bad["stream_capture_ready_score"] = 1
    with pytest.raises(ValueError):
        e.replay_value(bad)


def test_private_cli_and_terminal(tmp_path):
    public = tmp_path / "public.json"
    atomic_json(public, views())
    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--date", "20261004", "--fixture", str(public), "--output", str(output)]) == 0
    assert e.main(["--date", "20261004", "--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    value["completed_count"] -= 1
    atomic_json(output, value)
    assert e.main(["--date", "20261004", "--cold-replay", str(output)]) == 1


def private_inputs(tmp_path, monkeypatch):
    """Create authenticated private terminals; no real upstream bytes change."""
    refs = {}
    for role, view in views().items():
        p = tmp_path / (role + ".json")
        atomic_json(p, view)
        refs[role] = dict(path=str(p), sha256=sha256_file(p))
    names = [e.qualified.UPSTREAM, e.qualified.HISTORY]
    pins = {}
    for name in names:
        p = tmp_path / name
        terminal = p.parent / "raw" / p.stem / "terminal.json"
        value = dict(
            terminal_validation_sidecar_path=str(terminal),
            flagged_adversarial=False,
            cohort_ready_score=1,
            required_checks_passed=True,
            role_manifests=refs,
            gguf_sha256="fixture_protocol",
            model_revision="fixture_protocol",
        )
        atomic_json(p, value)
        digest = sha256_file(p)
        bound = terminal.parent / "bound.json"
        atomic_json(bound, dict(primary_sha256=digest, report=dict(passed=True)))
        atomic_json(
            terminal, dict(publication=dict(sidecar_path=str(bound), primary_sha256=digest))
        )
        pins[name] = digest
    monkeypatch.setattr(e, "INPUTS", [])
    monkeypatch.setattr(e.qualified, "PINS", pins)
    return refs


def test_preconditions_authenticate_or_block(tmp_path, monkeypatch):
    refs = private_inputs(tmp_path, monkeypatch)
    plan = e.preconditions(tmp_path)
    assert all(r["passed"] for r in plan["checks"])
    assert len(plan["slots"]) == 320
    Path(refs["stream"]["path"]).unlink()
    plan = e.preconditions(tmp_path)
    failed = [r for r in plan["checks"] if not r["passed"]]
    assert failed[0]["field"] == "public_sha256" and failed[0]["observed"] is None
    assert plan["slots"] == []
    upstream = tmp_path / e.qualified.UPSTREAM
    terminal = Path(json.loads(upstream.read_text())["terminal_validation_sidecar_path"])
    terminal.unlink()
    assert any(
        r["field"] == "authenticated_terminal" and not r["passed"]
        for r in e.preconditions(tmp_path)["checks"]
    )
    upstream.unlink()
    assert e.preconditions(tmp_path)["slots"] == []


def test_frozen_commands_and_qualified_wrappers(tmp_path, monkeypatch):
    specs = e.commands(tmp_path)
    names = {s["name"] for s in specs}
    assert {
        "cli_success",
        "cli_blocked",
        "cli_mutation",
        "cli_cold_replay",
        "changed_code_coverage",
    } <= names
    assert (
        next(s for s in specs if s["name"] == "repository_full_suite")["classification"]
        == "diagnostic"
    )
    monkeypatch.setattr(e.qualified, "validate", lambda *args: dict(passed=True))
    assert e.validate(tmp_path, tmp_path)["passed"]
    monkeypatch.setattr(e.qualified, "live_capture", lambda *args: dict(rows=[]))
    assert e.live_capture({}, tmp_path, tmp_path) == dict(rows=[])


@pytest.mark.parametrize("case", ["owned_fail", "cuda_block", "live", "upstream_block"])
def test_main_live_gates_in_private_storage(tmp_path, monkeypatch, case):
    slots = c.freeze(views())
    plan = dict(checks=[], references=[], slots=slots, capture_identity="private", manifests={})
    if case == "upstream_block":
        plan["checks"] = [e.operand("upstream", tmp_path / "missing", "exists", True, False)]
    monkeypatch.setattr(e, "preconditions", lambda root: plan)
    monkeypatch.setattr(
        e,
        "validate",
        lambda *args: dict(passed=case != "owned_fail", receipts=[], global_health=[]),
    )
    monkeypatch.setattr(
        e,
        "run_check",
        lambda *args, **kwargs: dict(
            passed=True,
            output_tail="CUDA" if case == "live" else "CPU",
            log_path=str(tmp_path / "cuda.log"),
            log_sha256="unused",
        ),
    )

    def live(plan, raw, scratch):
        ledger = c.prior.Ledger(raw / "ledger.json")
        return dict(
            rows=c.capture(slots, Runtime(), raw / "slots", "private", ledger=ledger),
            ledger=ledger.rows,
            checks=[],
        )

    monkeypatch.setattr(e, "live_capture", live)
    observed = {}
    monkeypatch.setattr(e, "terminal_publish", lambda output, value, raw: observed.update(value))
    assert e.main(["--date", "20261004", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    assert observed["verdict_class"] == (
        "disqualified" if case == "owned_fail" else "null" if case == "live" else "blocked"
    )
    assert observed["stream_capture_ready_score"] == 0


def test_terminal_failure_disqualifies_and_preserves_logs(tmp_path, monkeypatch):
    """REQ-REPORT-8102: a failed owned auditor cannot leave readiness enabled."""
    plan = dict(checks=[], references=[], slots=[], capture_identity="private", manifests={})
    value = e.build(
        plan,
        dict(rows=[], ledger=[], checks=[]),
        dict(passed=True, receipts=[]),
        tmp_path,
        0,
        fixture=True,
    )
    log = tmp_path / "auditor.log"
    log.write_text("private auditor failure\n")
    monkeypatch.setattr(
        e,
        "run_check",
        lambda *args, **kwargs: dict(passed=False, log_path=str(log), log_sha256=sha256_file(log)),
    )
    e.terminal_publish(tmp_path / (e.NAME + ".json"), value, tmp_path)
    assert value["verdict_class"] == "disqualified"
    assert value["flagged_adversarial"] is True
    assert value["required_checks_passed"] is False
    assert value["stream_capture_ready_score"] == 0


def test_original_floor_has_no_label_support_prerequisite(tmp_path):
    """REQ-VERIFY-8102: floors count sources, with masks at original positions."""
    slots = c.freeze(views())
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    for slot in slots:
        if slot["slot"] > c.FLOORS[slot["role"]]:
            slot.update(public_eligible=False, exclusion_reason="private_context_exclusion")
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger)
    reduced = c.reduce(rows)
    assert reduced["role_completion_counts"] == dict(stream=224, retention=48)
    assert reduced["stream_capture_ready_score"] == 1
    assert reduced["stream_features"][-1]["release_slot"] == 276
    rows[0].update(raw_response={}, status="failed")
    rows[0]["parsed"] = c.risk.transport.parse_response({}, rows[0]["visible_ids"])
    assert c.reduce(rows)["stream_capture_ready_score"] == 0
