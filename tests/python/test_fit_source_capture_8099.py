"""REQ-VERIFY-8099-CAPTURE; SCENARIO-VERIFY-8099-PRIVATE; REQ-REPORT-8099-PUBLICATION."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.verify import qwen_fit_source_capture_8099 as c
from carnot import experiment_8099_v701_fit_source_capture as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_8099_v701_fit_source_capture.py"


def views():
    """Only private public bytes enter the transport regression."""
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
                    order=i + (128 if role == "tune" else 0),
                )
                for i in range(n)
            ],
        )
        for role, n in c.ROLES.items()
    }


class Runtime:
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


def test_frozen_control_contract():
    slots = c.freeze(views())
    assert len(slots) == 288
    assert c.config()["output_tokens"] == 27648
    assert [s["unit_id"] for s in slots[:128]] == [f"fit-{i}" for i in range(128)]
    by = {(s["unit_id"], s["arm"]): s for s in slots}
    for i in range(32):
        full = by[(f"fit-{i}", "full_source")]
        assert full["request"] == by[(f"fit-{i}", "duplicate")]["request"]
        for arm in c.ARMS:
            body = json.loads(by[(f"fit-{i}", arm)]["request"]["messages"][1]["content"])
            assert body["original_answer"] == "Answer unchanged."
            assert "y" not in body
        assert (
            json.loads(by[(f"fit-{i}", "source_removed")]["request"]["messages"][1]["content"])[
                "complete_source"
            ]
            == ""
        )
        assert (
            json.loads(by[(f"fit-{i}", "source_mismatched")]["request"]["messages"][1]["content"])[
                "complete_source"
            ]
            == f"Source fit {(i + 1) % 32}."
        )
    for bad in [{}, {"fit": views()["fit"]}]:
        with pytest.raises(ValueError):
            c.freeze(bad)
    bad = views()
    bad["fit"]["roster"].pop()
    with pytest.raises(ValueError):
        c.freeze(bad)
    bad = views()
    bad["fit"]["request_rows"][0]["y"] = 1
    with pytest.raises(ValueError):
        c.freeze(bad)
    bad = views()
    bad["fit"]["roster"][0]["label"] = 1
    with pytest.raises(ValueError):
        c.freeze(bad)
    bad = views()
    bad["tune"]["roster"][0]["source_cluster_id"] = "cluster-fit-0"
    with pytest.raises(ValueError):
        c.freeze(bad)
    bad = views()
    bad["fit"]["roster"][0]["role"] = "tune"
    with pytest.raises(ValueError):
        c.freeze(bad)


def test_null_capture_and_replay(tmp_path):
    slots = c.freeze(views())
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger)
    reduced = c.reduce(rows)
    assert reduced["fit_capture_ready_score"] == 1
    assert reduced["independent_count"] == 192
    assert reduced["completed_count"] == 288
    assert reduced["duplicate_variation"]["mean_absolute_difference"] == 0
    assert reduced["source_effect_diagnostics"]["source_removed"]["paired_count"] == 32
    assert all(r["human_target"] is None for r in rows if r["arm"] != "full_source")
    assert c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger) == rows
    with pytest.raises(ValueError):
        c.reduce(rows + rows[:1])
    bad = deepcopy(rows)
    bad[0]["parsed"]["probability"] = 1
    with pytest.raises(ValueError):
        c.reduce(bad)
    with pytest.raises(ValueError):
        c.capture(slots, Runtime(), tmp_path / "slots", "changed", ledger=ledger)


def test_interrupted_and_budget_slots(tmp_path):
    slots = c.freeze(views())[:2]
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    ledger.start("generation", slots[0]["family_id"], slots[0]["request"])
    rows = c.capture(slots, Runtime(), tmp_path / "slots", "private", ledger=ledger, deadline_s=0)
    assert rows[0]["failure_kind"] == "interrupted"
    assert rows[1]["status"] == "censored"
    assert ledger.counts()["generation_calls_attempted"] == 1
    assert c.reduce(rows)["fit_capture_ready_score"] == 0
    rows = c.capture(
        c.freeze(views())[:1],
        Runtime(),
        tmp_path / "blocked",
        "private",
        ledger=c.prior.Ledger(tmp_path / "blocked-ledger.json"),
        blocked_reason="missing_cuda",
    )
    assert rows[0]["exclusion_reason"] == "missing_cuda"
    assert not rows[0]["started"]


@pytest.mark.parametrize("mode", ["timeout", "invalid", "truncated", "context", "tokenizer"])
def test_malformed_transport(tmp_path, mode):
    class Bad(Runtime):
        def count(self, text):
            if mode == "tokenizer":
                raise ValueError("broken")
            return 7000 if mode == "context" else 30

        def generate(self, request):
            if mode == "timeout":
                raise TimeoutError("private")
            value = super().generate(request)
            if mode == "invalid":
                value["choices"][0]["message"]["content"] = (
                    '{"unsupported_probability":2,"source_sentence_id":null}'
                )
            if mode == "truncated":
                value["choices"][0]["finish_reason"] = "length"
            return value

    rows = c.capture(
        c.freeze(views())[:1],
        Bad(),
        tmp_path / "slots",
        "private",
        ledger=c.prior.Ledger(tmp_path / "ledger.json"),
    )
    assert rows[0]["numerator"] == 0
    assert (
        rows[0]["failure_kind"]
        == {
            "timeout": "timeout",
            "invalid": "invalid_probability",
            "truncated": "truncation",
            "context": "context",
            "tokenizer": "transport",
        }[mode]
    )
    assert c.reduce(rows)["completed_count"] == 0


def test_real_cli_publication_block_mutation_and_replay(tmp_path):
    """SCENARIO-REPORT-8099-CLI uses external CWD and private primaries."""
    fixture = tmp_path / "public.json"
    atomic_json(fixture, views())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"

    def run(*args):
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(CLI), "--date", "20261004", *map(str, args)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result

    output = tmp_path / "success" / (e.NAME + ".json")
    assert run("--fixture", fixture, "--output", output).returncode == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["fit_capture_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert run("--cold-replay", output).returncode == 0
    value["completed_count"] += 1
    atomic_json(output, value)
    assert run("--cold-replay", output).returncode == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert run("--fixture", tmp_path / "absent", "--output", blocked).returncode == 0
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    bad = views()
    bad["fit"]["request_rows"][0]["labels"] = []
    atomic_json(fixture, bad)
    mutated = tmp_path / "mutated" / (e.NAME + ".json")
    assert run("--fixture", fixture, "--output", mutated).returncode == 0
    assert json.loads(mutated.read_text())["verdict_class"] == "disqualified"
    assert run("--date", "bad").returncode == 2


def test_build_validation_and_raw_mutations(tmp_path):
    plan = dict(checks=[], references=[], slots=c.freeze(views()), protocol={}, manifests={})
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(plan["slots"], Runtime(), tmp_path / "slots", "private", ledger=ledger)
    result = dict(
        rows=rows,
        ledger=ledger.rows,
        checks=[],
        measured_duration_s=12,
        model_identity_receipt=dict(
            authenticated=True, offload_layers=[1, 1], props=dict(chat_template="private")
        ),
        gpu_lease_receipt=dict(owner_pid=os.getpid()),
        runtime_receipts=[],
    )
    validation = dict(passed=True, receipts=[])
    value = e.build(plan, result, validation, tmp_path, 12, fixture=True)
    assert value["fit_capture_ready_score"] == 0
    assert e.replay_value(value)
    live = e.build(plan, result, validation, tmp_path, 12, fixture=False)
    assert live["fit_capture_ready_score"] == 1
    bad = e.build(plan, result, dict(passed=False, receipts=[]), tmp_path, 12, fixture=False)
    assert bad["verdict_class"] == "disqualified"
    for key in [
        "completed_count",
        "independent_count",
        "eligible_count",
        "fit_capture_ready_score",
    ]:
        changed = deepcopy(value)
        changed[key] += 1
        with pytest.raises(ValueError):
            e.replay_value(changed)
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError):
        e.replay_value(changed)


def test_preconditions_and_live_adapter(tmp_path, monkeypatch):
    """Missing external evidence must report the exact failed operand."""
    plan = e.preconditions(tmp_path)
    assert plan["checks"] and all("observed" in r for r in plan["checks"])
    assert not plan["slots"]
    frozen = c.freeze(views())

    def fake(plan, raw, scratch):
        assert e.legacy.capture.freeze({}) == frozen
        runtime = e.legacy.QwenRuntime(Path("private"), scratch, 0)
        runtime.load()
        assert runtime.receipts == []
        return dict(
            rows=e.legacy.capture.capture(frozen, Runtime(), raw / "slots", "private"),
            checks=[e.operand("private_cuda", Path("private"), "owned", True, True)],
        )

    monkeypatch.setattr(e.legacy, "live_capture", fake)
    monkeypatch.setattr(e.legacy.QwenRuntime, "load", lambda self: dict(authenticated=True))
    result = e.live_capture(
        dict(slots=frozen, manifests={}, protocol={}, capture_identity="private"),
        tmp_path,
        tmp_path,
    )
    assert result["ledger"][0]["status"] == "completed"

    def fail(self):
        raise RuntimeError("private load failure")

    monkeypatch.setattr(e.legacy.QwenRuntime, "load", fail)
    with pytest.raises(RuntimeError):
        e.live_capture(
            dict(slots=frozen, manifests={}, protocol={}, capture_identity="private"),
            tmp_path / "failure",
            tmp_path,
        )


def test_validation_manifest_and_private_main_routes(tmp_path, monkeypatch):
    """REQ-REPORT-8099-PUBLICATION freezes checks before starting model work."""
    specs = e.commands(tmp_path)
    assert any(
        r["name"] == "repository_full_suite" and r["classification"] == "diagnostic" for r in specs
    )
    events = []

    def check(root, spec, private, durable, **kwargs):
        events.append(spec["name"])
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("CUDA private test observation")
        return dict(
            spec,
            passed=True,
            log_path=str(log),
            log_sha256=sha256_file(log),
            output_tail="CUDA private test observation",
            exit_code=0,
            timed_out=False,
        )

    monkeypatch.setattr(e, "run_check", check)
    validation = e.validate(tmp_path / "raw", tmp_path)
    assert validation["passed"] and len(validation["global_health"]) == 1
    slots = c.freeze(views())
    plan = dict(
        checks=[], references=[], slots=slots, protocol={}, manifests={}, capture_identity="private"
    )
    monkeypatch.setattr(e, "preconditions", lambda root: deepcopy(plan))
    monkeypatch.setattr(
        e, "validate", lambda raw, private: dict(passed=True, receipts=[], global_health=[])
    )

    def live(plan, raw, scratch):
        ledger = c.prior.Ledger(raw / "ledger.json")
        return dict(
            rows=c.capture(slots, Runtime(), raw / "slots", "private", ledger=ledger),
            ledger=ledger.rows,
            checks=[e.operand("private_cuda", Path("private"), "owned", True, True)],
            measured_duration_s=12,
            model_identity_receipt=dict(authenticated=True, offload_layers=[1, 1]),
            gpu_lease_receipt=dict(owner_pid=os.getpid()),
        )

    monkeypatch.setattr(e, "live_capture", live)
    output = tmp_path / "live" / (e.NAME + ".json")
    assert e.main(["--date", "20261004", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["fit_capture_ready_score"] == 1
    assert e.main(["--date", "20261004", "--cold-replay", str(output)]) == 0
    assert e.main(["--date", "20261004", "--cold-replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(
        e, "validate", lambda raw, private: dict(passed=False, receipts=[], global_health=[])
    )
    output = tmp_path / "owned_failure" / (e.NAME + ".json")
    assert e.main(["--date", "20261004", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    fixture = tmp_path / "public.json"
    atomic_json(fixture, views())

    def rejected(root, spec, private, durable, **kwargs):
        receipt = check(root, spec, private, durable, **kwargs)
        receipt["passed"] = False
        return receipt

    monkeypatch.setattr(e, "run_check", rejected)
    output = tmp_path / "terminal_failure" / (e.NAME + ".json")
    assert e.main(["--date", "20261004", "--fixture", str(fixture), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"


def test_terminal_authentication_public_only(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8099-CLI never opens evaluator labels in the worker."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in e.INPUTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private")
    refs = {}
    for role, view in views().items():
        path = root / (role + ".json")
        atomic_json(path, view)
        refs[role] = dict(path=str(path), sha256=sha256_file(path))
    pins = {}
    for name in (e.UPSTREAM, e.HISTORY):
        path = root / name
        raw = path.parent / "raw" / path.stem
        terminal = raw / "terminal.json"
        value = dict(
            experiment_id=8098 if name == e.UPSTREAM else 7995,
            task_id="exp8098-private" if name == e.UPSTREAM else "exp7995-private",
            honest_verdict="complete_null_private",
            verdict_class="null",
            flagged_adversarial=False,
            cohort_ready_score=1,
            fit_support_ready_score=1,
            required_checks_passed=True,
            role_manifests=refs,
            terminal_validation_sidecar_path=str(terminal),
            gguf_sha256="private",
            model_revision="private",
        )
        publication = e.publish_primary(path, value, lambda _: dict(passed=True))
        atomic_json(terminal, dict(publication=publication) if name == e.UPSTREAM else publication)
        pins[name] = sha256_file(path)
    monkeypatch.setattr(e, "PINS", pins)
    plan = e.preconditions(root)
    assert len(plan["slots"]) == 288 and all(r["passed"] for r in plan["checks"])
    Path(refs["fit"]["path"]).write_text("{}")
    assert not e.preconditions(root)["slots"]
    terminal.unlink()
    assert any(
        r["field"] == "authenticated_terminal" and not r["passed"]
        for r in e.preconditions(root)["checks"]
    )


def test_replay_rejects_raw_ledger_and_check_drift(tmp_path):
    slots = c.freeze(views())
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(slots[:1], Runtime(), tmp_path / "slots", "private", ledger=ledger)
    plan = dict(checks=[], references=[], slots=slots, protocol={}, manifests={})
    log = tmp_path / "check.log"
    log.write_text("current")
    value = e.build(
        plan,
        dict(rows=rows, ledger=ledger.rows, checks=[]),
        dict(passed=True, receipts=[dict(log_path=str(log), log_sha256=sha256_file(log))]),
        tmp_path,
        1,
        fixture=True,
    )
    changed = deepcopy(value)
    changed["model_invocation_counts"]["generation_calls_attempted"] += 1
    with pytest.raises(ValueError, match="current_receipt_drift"):
        e.replay_value(changed)
    log.write_text("changed")
    with pytest.raises(ValueError, match="validation_log_drift"):
        e.replay_value(value)
    atomic_json(Path(value["raw_shard_hashes"][0]["path"]), dict(rows=[]))
    value["raw_shard_hashes"][0]["sha256"] = sha256_file(Path(value["raw_shard_hashes"][0]["path"]))
    with pytest.raises(ValueError, match="raw_rows_drift"):
        e.replay_value(value)
    ledger.rows[0]["status"] = "completed"
    (tmp_path / "slots" / "slot-000.json").unlink()
    with pytest.raises(ValueError, match="lost_raw_receipt"):
        c.capture(slots[:1], Runtime(), tmp_path / "slots", "private", ledger=ledger)


def test_private_direct_fixture_mutation(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8099-CLI rejects label access before a model is loaded."""
    fixture = tmp_path / "mutated.json"
    bad = views()
    bad["fit"]["request_rows"][0]["y"] = 1
    atomic_json(fixture, bad)

    def check(root, spec, private, durable, **kwargs):
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private normal exit")
        return dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))

    monkeypatch.setattr(e, "run_check", check)
    assert (
        e.main(
            [
                "--date",
                "20261004",
                "--fixture",
                str(fixture),
                "--output",
                str(tmp_path / "mutated" / (e.NAME + ".json")),
            ]
        )
        == 0
    )


def test_independent_reduction_mutation_and_check_scope(tmp_path, monkeypatch):
    """REQ-REPORT-8099-PUBLICATION checks independently counted denominators."""
    slots = c.freeze(views())
    ledger = c.prior.Ledger(tmp_path / "ledger.json")
    rows = c.capture(slots[:1], Runtime(), tmp_path / "slots", "private", ledger=ledger)
    plan = dict(checks=[], references=[], slots=slots, protocol={}, manifests={})
    value = e.build(
        plan,
        dict(rows=rows, ledger=ledger.rows, checks=[]),
        dict(passed=True, receipts=[]),
        tmp_path,
        1,
        fixture=True,
    )
    value["completed_count"] += 1
    with pytest.raises(ValueError, match="independent_reduction_drift"):
        e.replay_value(value)

    def fake(plan, raw, scratch):
        return dict(rows=[], checks=[dict(upstream_id="cuda", field="observed", passed=True)])

    monkeypatch.setattr(e.legacy, "live_capture", fake)
    assert (
        e.live_capture(plan, tmp_path / "adapter", tmp_path)["checks"][0]["check"]
        == "cuda_observed"
    )


def test_validation_keeps_global_health_and_coverage(tmp_path, monkeypatch):
    """REQ-REPORT-8099-PUBLICATION keeps a global failure out of scientific gates."""
    raw = tmp_path / "raw" / e.NAME / "invocations" / "private"
    raw.mkdir(parents=True)
    calls = []

    def check(root, spec, private, durable, **kwargs):
        calls.append(spec["name"])
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private")
        return dict(spec, passed=True, log_path=str(log), log_sha256=sha256_file(log))

    monkeypatch.setattr(e, "run_check", check)
    atomic_json(tmp_path / "coverage.json", dict(files={}, totals=dict(percent_covered=100)))
    assert e.validate(raw, tmp_path)["passed"]
    assert (raw / "coverage.json").exists()
    calls.clear()
    assert e.validate(raw, tmp_path)["passed"]
    assert "repository_full_suite" not in calls
    saved = json.loads((raw.parent / "global_health.json").read_text())
    Path(saved["receipt"]["log_path"]).write_text("mutated")
    with pytest.raises(ValueError, match="global_health_log_drift"):
        e.validate(raw, tmp_path)


def test_no_call_block_declares_no_inference(tmp_path):
    """REQ-REPORT-8099-PUBLICATION must not claim inference when admission blocks."""
    plan = dict(
        checks=[e.operand("missing_cuda", tmp_path, "available", True, False)],
        references=[],
        slots=[],
        protocol={},
        manifests={},
    )
    value = e.build(
        plan,
        dict(rows=[], ledger=[], checks=[]),
        dict(passed=True, receipts=[]),
        tmp_path,
        1,
        fixture=False,
    )
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "no_model_load"
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert value["MODEL_SPECS"] == []
